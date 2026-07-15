import os
import subprocess

from dotenv import load_dotenv
from funasr import AutoModel
from groq import Groq
from pydub import AudioSegment

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------------------------
# Config
# ------------------------------------

# Groq's free tier caps uploads at 25MB (100MB on the dev/paid tier).
# We stay comfortably under that so re-encoding overhead doesn't push
# a chunk back over the limit.
MAX_CHUNK_BYTES = 20 * 1024 * 1024  # 20MB safety margin

# If no sentence-ending punctuation shows up within this many words,
# force a break anyway. Guards against run-on speech / missed punctuation
# producing one giant "sentence".
MAX_SENTENCE_WORDS = 50

# Common abbreviations that end in "." but do NOT mark a sentence end.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "st", "no", "vol", "fig", "approx",
    "gen", "capt", "col", "sgt", "lt", "rev", "inc", "ltd", "co",
}

# ------------------------------------
# Segment-merging config (sentence -> final segment)
# ------------------------------------

# A final "segment" (the unit stored/embedded downstream) is built by
# greedily merging complete sentences together until the segment's
# running duration reaches this many seconds.
SEGMENT_MIN_SECONDS = 20.0

# While merging, if the gap between the current segment's last sentence
# and the next sentence is at least this many seconds, the segment is
# closed early -- even if it hasn't reached SEGMENT_MIN_SECONDS yet --
# rather than reaching across a long pause to hit the duration target.
SEGMENT_MAX_GAP_SECONDS = 10.0

# ------------------------------------
# Audio stretch config
# ------------------------------------

# atempo_factor = raw_audio_dur / video_dur. A factor of 1.0 means no
# drift at all. Anything beyond +/-3% drift usually means something is
# actually wrong with the source stream (dropped frames, VFR video,
# corrupt container) rather than ordinary clock drift, so we stop
# instead of silently stretching audio by a large, suspicious amount.
MAX_STRETCH_DRIFT = 0.03  # 3%

# ------------------------------------
# FunASR (fsmn-vad) silence detection config
# ------------------------------------

# Only keep silence gaps at least this long.
MIN_SILENCE_SECONDS = 10

# Used twice in _detect_silence_funasr, for two different merge passes
# with the same "close enough to be one continuous thing" intent:
#   1. Before computing gaps: merge raw VAD speech segments separated by
#      less than this many seconds, so a short in-speech pause doesn't
#      get mistaken for a stand-alone silence.
#   2. After computing the final silence list: merge two silence regions
#      that ended up separated by an isolated stretch of "speech" shorter
#      than this many seconds (e.g. a single stray word or cough sitting
#      between two long silences). Pass (1) only merges speech segments
#      that are themselves close together -- it does NOT catch this case,
#      since an isolated blip far from any other speech has nothing
#      nearby to merge with during pass (1), yet still splits what should
#      be reported as one continuous silence into two.
MERGE_GAP_SECONDS = 2

_vad_model = None  # loaded lazily, once


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = AutoModel(model="fsmn-vad")
    return _vad_model


# ------------------------------------
# Small helpers (duration probing, formatting)
# ------------------------------------
def _get_duration(path_or_url: str, stream: str | None = None) -> float:
    """
    Returns duration in seconds. If `stream` is given (e.g. "v:0" or
    "a:0"), reads that stream's own duration field for precision;
    falls back to container-level format duration if the stream
    doesn't expose one.
    """
    if stream:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", stream,
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path_or_url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        value = r.stdout.strip()
        if r.returncode == 0 and value and value != "N/A":
            return float(value)

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path_or_url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{r.stderr}")
    return float(r.stdout.strip())


def _seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _build_atempo_chain(factor: float) -> str:
    """
    ffmpeg's atempo filter only accepts 0.5-2.0 in a single pass; chain
    multiple atempo filters if the factor ever falls outside that.
    """
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor}"
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining}")
    return ",".join(filters)


# ------------------------------------
# Audio extraction + time-stretch to match video duration
# ------------------------------------
def _extract_and_stretch_audio(
    video_url: str,
    raw_audio_path: str = "audio_raw.wav",
    stretched_audio_path: str = "audio.wav",
    keep_raw_audio: bool = False,
) -> str:
    """
    Extracts audio directly from video_url via ffmpeg, then time-stretches
    it (pitch-preserving) so its duration exactly matches the video's own
    duration. This keeps transcript timestamps mapped 1:1 onto the video,
    which otherwise drifts on long recordings due to encoder clock skew.

    Prints video length, raw audio length, and the computed stretch
    factor. Raises if the drift implied by that factor is larger than
    MAX_STRETCH_DRIFT, since that usually signals a broken source stream
    rather than ordinary clock drift.
    """
    extract_command = [
        "ffmpeg",
        "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-i", video_url,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        raw_audio_path,
    ]

    print("Extracting audio from URL...")
    extract_result = subprocess.run(extract_command, capture_output=True, text=True)
    if extract_result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed:\n{extract_result.stderr}")

    print("Probing durations...")
    video_dur = _get_duration(video_url, stream="v:0")
    raw_audio_dur = _get_duration(raw_audio_path, stream="a:0")
    atempo_factor = raw_audio_dur / video_dur
    drift_pct = abs(atempo_factor - 1.0)

    print("\n--- Stretch indicator -----------------------------")
    print(f"Video length:      {video_dur:.2f}s  ({_seconds_to_hms(video_dur)})")
    print(f"Raw audio length:  {raw_audio_dur:.2f}s  ({_seconds_to_hms(raw_audio_dur)})")
    print(f"Drift before stretch: {video_dur - raw_audio_dur:+.2f}s")
    print(f"Computed atempo factor: {atempo_factor:.6f}  ({drift_pct * 100:.2f}% drift)")
    print("-----------------------------------------------------\n")

    if drift_pct > MAX_STRETCH_DRIFT:
        raise RuntimeError(
            f"Stretch factor {atempo_factor:.6f} implies {drift_pct * 100:.2f}% drift, "
            f"which exceeds the {MAX_STRETCH_DRIFT * 100:.0f}% safety limit. "
            f"This usually means the source stream itself is broken (dropped frames, "
            f"VFR video, corrupt container) rather than ordinary clock drift -- "
            f"stretching audio by this much would badly desync the transcript."
        )

    atempo_chain = _build_atempo_chain(atempo_factor)

    stretch_command = [
        "ffmpeg",
        "-y",
        "-i", raw_audio_path,
        "-filter:a", atempo_chain,
        "-ar", "16000",
        "-ac", "1",
        stretched_audio_path,
    ]

    print("Time-stretching audio to match video duration...")
    stretch_result = subprocess.run(stretch_command, capture_output=True, text=True)
    if stretch_result.returncode != 0:
        raise RuntimeError(f"ffmpeg stretch failed:\n{stretch_result.stderr}")

    final_audio_dur = _get_duration(stretched_audio_path, stream="a:0")
    print(f"Stretched audio length: {final_audio_dur:.2f}s  ({_seconds_to_hms(final_audio_dur)})")
    print(f"Remaining drift: {video_dur - final_audio_dur:+.3f}s\n")

    if not keep_raw_audio and os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)

    return stretched_audio_path


# ------------------------------------
# Chunking helper (for Whisper's upload size limit -- NOT the final
# transcript "segments"; this only exists to satisfy Groq's per-request
# upload cap and is deleted again right after transcription)
# ------------------------------------
def _split_audio(file_path: str, chunk_dir: str = "_chunks"):
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)
    total_bytes = os.path.getsize(file_path)

    bytes_per_ms = total_bytes / total_ms
    chunk_ms = max(1000, int(MAX_CHUNK_BYTES / bytes_per_ms))

    chunks = []
    for i, start_ms in enumerate(range(0, total_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, total_ms)
        piece = audio[start_ms:end_ms]

        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.wav")
        piece.export(chunk_path, format="wav")
        chunks.append((chunk_path, start_ms / 1000.0))

    return chunks, chunk_dir


# ------------------------------------
# FunASR (fsmn-vad) silence detection
# ------------------------------------
def _merge_close_silences(silences: list[dict], max_gap: float) -> list[dict]:
    """
    Merges consecutive silence regions whose gap -- the stretch of
    "speech" sitting between them -- is shorter than `max_gap` seconds
    into a single silence region spanning the whole run.

    This exists on top of the speech-merge pass inside
    _detect_silence_funasr(): that earlier pass only merges raw VAD
    speech segments that are themselves close together, so it can't
    catch an ISOLATED short speech blip (e.g. a single stray word or
    cough) sitting far from any other speech, in between two otherwise
    long silences -- that blip has nothing nearby to merge with during
    the speech pass, yet still splits what should be reported as one
    continuous silence into two separate entries.

    The small gap itself is swallowed, not preserved: the merged range
    is [first.start, last.end], with `duration` recomputed from that.
    No rounding is applied here, matching the full-precision timestamps
    the rest of this module reports.

    Silences are sorted by start first since callers may pass them in
    any order.
    """
    if not silences:
        return []

    ordered = sorted(silences, key=lambda s: s["start"])
    merged = [dict(ordered[0])]

    for s in ordered[1:]:
        last = merged[-1]
        gap = s["start"] - last["end"]
        if gap < max_gap:
            last["end"] = max(last["end"], s["end"])
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(dict(s))

    return merged


def _detect_silence_funasr(
    audio_path: str,
    total_duration: float,
    min_silence_seconds: float = MIN_SILENCE_SECONDS,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
) -> list[dict]:
    """
    Runs FunASR's fsmn-vad model to find every speech segment in the
    file, merges segments separated by less than merge_gap_seconds,
    then reports every remaining gap (including leading silence before
    the first speech and trailing silence after the last) that is at
    least min_silence_seconds long.

    Before returning, any two of those reported silence regions that
    ended up separated by less than merge_gap_seconds of intervening
    "speech" are merged into one, swallowing that short gap (see
    _merge_close_silences()).

    Silence boundaries are reported at full precision (no rounding) --
    the exact start/end/duration the VAD model produced.
    """
    model = _get_vad_model()

    print("Running FunASR fsmn-vad...")
    result = model.generate(input=audio_path)

    speech_segments = [
        {"start": seg[0] / 1000, "end": seg[1] / 1000}
        for seg in result[0]["value"]
    ]
    print(f"Speech segments (raw): {len(speech_segments)}")

    merged = []
    for seg in speech_segments:
        if not merged:
            merged.append(seg)
            continue
        if seg["start"] - merged[-1]["end"] <= merge_gap_seconds:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)
    print(f"Speech segments (merged): {len(merged)}")

    def _make_silence(start: float, end: float) -> dict | None:
        if end <= start:
            return None
        return {
            "start": start,
            "end": end,
            "duration": end - start,
        }

    silence = []
    prev_end = 0.0

    for seg in merged:
        gap = seg["start"] - prev_end
        if gap >= min_silence_seconds:
            entry = _make_silence(prev_end, seg["start"])
            if entry:
                silence.append(entry)
        prev_end = seg["end"]

    # Trailing silence, always checked against total_duration so it's
    # captured even if merged speech segments run right up near the end.
    tail_gap = total_duration - prev_end
    if tail_gap >= min_silence_seconds:
        entry = _make_silence(prev_end, total_duration)
        if entry:
            silence.append(entry)

    print(f"Silence regions (>= {min_silence_seconds}s): {len(silence)}")

    before_merge = len(silence)
    silence = _merge_close_silences(silence, merge_gap_seconds)
    if len(silence) != before_merge:
        print(
            f"Merged silence regions separated by < {merge_gap_seconds}s of "
            f"intervening speech: {before_merge} -> {len(silence)}"
        )

    return silence


def _overlaps_silence(word: dict, silence_zones: list[dict]) -> bool:
    """
    True if the word's [start, end] interval touches a silence zone at
    all -- including merely sharing an endpoint -- not just if its
    midpoint falls inside one. This is intentionally aggressive: any
    word whose audio even brushes a silence region gets dropped.
    """
    return any(word["start"] <= z["end"] and word["end"] >= z["start"] for z in silence_zones)


# ------------------------------------
# Sentence-boundary helpers (word-level rebuild)
# ------------------------------------
def _is_sentence_end(word_text: str, next_word_text: str | None) -> bool:
    stripped = word_text.rstrip("\"')]»”’")
    if not stripped or stripped[-1] not in ".!?":
        return False

    if stripped[-1] == ".":
        core = stripped[:-1].lower()
        if core in _ABBREVIATIONS:
            return False
        if len(core) <= 2 and core.isalpha():
            return False

    if next_word_text:
        first_alpha = next((c for c in next_word_text if c.isalpha()), None)
        if first_alpha is not None and first_alpha.islower():
            return False

    return True


def _gap_overlaps_silence(gap_start: float, gap_end: float, silence_zones: list[dict]) -> bool:
    """
    True if any silence zone overlaps the (gap_start, gap_end) interval
    that sits between two consecutive surviving words. Used to force a
    sentence cut even when no sentence-ending punctuation was found, so
    a silence region can never end up sitting inside a single sentence.
    """
    return any(z["start"] < gap_end and z["end"] > gap_start for z in silence_zones)


def _words_to_sentences(words: list[dict], silence_zones: list[dict] | None = None) -> list[dict]:
    """
    Rebuilds SENTENCE-level units from word-level timestamps (this is an
    intermediate step -- the final "segments" returned by ingestion()
    are built by merging several of these sentences together, see
    _sentences_to_segments()).

    A sentence is flushed when any of the following happens:
      - natural sentence-ending punctuation is found,
      - the running word count hits MAX_SENTENCE_WORDS, or
      - a FunASR silence zone falls in the gap between the current word
        and the next one, guaranteeing a silence region is never left
        sitting inside a single sentence's [start, end] span.
    """
    silence_zones = silence_zones or []
    sentences = []
    current_words: list[dict] = []
    seg_id = 0

    def _flush():
        nonlocal seg_id
        if not current_words:
            return
        start = current_words[0]["start"]
        end = current_words[-1]["end"]
        text = " ".join(w["text"] for w in current_words).strip()
        sentences.append(
            {
                "id": seg_id,
                "start": start,
                "end": end,
                "text": text,
            }
        )
        seg_id += 1

    for i, w in enumerate(words):
        current_words.append(w)
        next_word = words[i + 1] if i + 1 < len(words) else None
        next_text = next_word["text"] if next_word else None

        sentence_ends_naturally = _is_sentence_end(w["text"], next_text)
        hit_word_cap = len(current_words) >= MAX_SENTENCE_WORDS
        silence_forces_cut = next_word is not None and _gap_overlaps_silence(
            w["end"], next_word["start"], silence_zones
        )

        if sentence_ends_naturally or hit_word_cap or silence_forces_cut:
            _flush()
            current_words = []

    _flush()
    return sentences


# ------------------------------------
# Sentence -> final segment merging (duration-based)
# ------------------------------------
def _sentences_to_segments(
    sentences: list[dict],
    min_segment_seconds: float = SEGMENT_MIN_SECONDS,
    max_gap_seconds: float = SEGMENT_MAX_GAP_SECONDS,
) -> list[dict]:
    """
    Greedily merges consecutive complete SENTENCES (from
    _words_to_sentences()) into final SEGMENTS -- the unit this module
    actually returns and that downstream chunking/embedding operates on.

    Rule
    ----
    Keep appending whole sentences to the current segment while its
    running duration (last_sentence.end - first_sentence.start) is still
    under `min_segment_seconds` (20s by default). As soon as adding a
    sentence pushes the running duration to >= min_segment_seconds, the
    segment is closed right there (including the sentence that pushed it
    over) and a new segment starts with the next sentence.

    Early close override
    ---------------------
    While still under the duration threshold, if the time gap between
    the current segment's last sentence and the next sentence
    (next_sentence["start"] - current_segment_last_sentence["end"]) is
    >= max_gap_seconds (10s by default), the segment is closed early --
    even though it hasn't reached 20s yet -- rather than reaching across
    a long pause just to hit the duration target. The next sentence
    starts a fresh segment.

    A single sentence that is already >= min_segment_seconds long on its
    own (e.g. it hit MAX_SENTENCE_WORDS during a long silence-free run)
    becomes its own one-sentence segment immediately.

    The very last segment of the transcript is kept as-is even if it
    never reaches min_segment_seconds -- it simply runs out of sentences
    to merge into it; it is not padded, merged backward, or dropped.

    Returns
    -------
    List of segment dicts: {"id", "start", "end", "duration", "text"},
    "id" being a fresh 0-indexed, contiguous counter over the final
    segments (independent of the sentence-level "id"s).
    """
    if not sentences:
        return []

    def _duration(sents: list[dict]) -> float:
        return sents[-1]["end"] - sents[0]["start"]

    seg_id = 0
    segments = []

    def _flush(sents: list[dict]):
        nonlocal seg_id
        start = sents[0]["start"]
        end = sents[-1]["end"]
        text = " ".join(s["text"].strip() for s in sents).strip()
        segments.append(
            {
                "id": seg_id,
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "text": text,
            }
        )
        seg_id += 1

    current = [sentences[0]]

    for nxt in sentences[1:]:
        if _duration(current) >= min_segment_seconds:
            # Already hit the duration target with what's accumulated so
            # far -- close now, `nxt` starts a brand new segment.
            _flush(current)
            current = [nxt]
            continue

        gap = nxt["start"] - current[-1]["end"]
        if gap >= max_gap_seconds:
            # Next sentence is too far away -- close early rather than
            # merging across a long pause, even though we're still under
            # the duration threshold.
            _flush(current)
            current = [nxt]
            continue

        current.append(nxt)

    if current:
        _flush(current)

    return segments


# ------------------------------------
# Transcription + FunASR-based cleanup
# ------------------------------------
def ingestion(
    video_url: str,
    raw_audio_path: str = "audio_raw.wav",
    audio_path: str = "audio.wav",
    keep_audio: bool = True,
    min_silence_seconds: float = MIN_SILENCE_SECONDS,
    min_segment_seconds: float = SEGMENT_MIN_SECONDS,
    max_segment_gap_seconds: float = SEGMENT_MAX_GAP_SECONDS,
):
    """
    1. Extracts audio from video_url and time-stretches it to exactly
       match the video's duration (prints video/raw-audio length and
       the stretch factor; raises if drift exceeds MAX_STRETCH_DRIFT).
    2. Runs FunASR fsmn-vad on the final stretched audio to find every
       silence region >= min_silence_seconds (including leading and
       trailing silence), merging silence regions separated by an
       isolated short speech blip, and reports the result at full
       precision (no rounding).
    3. Transcribes the same audio with Groq Whisper (word-level
       timestamps, chunked if the file is too large for one upload).
    4. Drops every transcribed word that touches a FunASR silence
       region at all -- any overlap, even just sharing an endpoint --
       not only words whose midpoint falls inside one.
    5. Regroups surviving words into SENTENCE-level units, forcing a
       sentence break wherever a silence zone falls in the gap between
       two consecutive surviving words.
    6. Merges consecutive sentences into final duration-based SEGMENTS:
       a segment keeps growing until it reaches min_segment_seconds
       (20s default), closing early if the gap to the next sentence is
       >= max_segment_gap_seconds (10s default). See
       _sentences_to_segments() for the exact rule.

    Returns
    -------
    {
        "segments": [...],       # final duration-based transcript segments
        "text": "...",           # every segment's text joined into one
                                  # single paragraph -- the entire
                                  # transcript as one block, in addition
                                  # to the per-segment breakdown above
        "max_duration": ...,
        "silences": [...],       # FunASR-detected silence zones
    }
    """
    _extract_and_stretch_audio(video_url, raw_audio_path, audio_path, keep_raw_audio=False)

    total_duration = _get_duration(audio_path, stream="a:0")

    # Silence detection runs on the final (stretched) audio, so its
    # timestamps line up exactly with the transcript we're about to
    # produce from the same file.
    silence_zones = _detect_silence_funasr(audio_path, total_duration, min_silence_seconds)

    file_size = os.path.getsize(audio_path)

    chunk_dir = None
    if file_size <= MAX_CHUNK_BYTES:
        chunk_list = [(audio_path, 0.0)]
    else:
        chunk_list, chunk_dir = _split_audio(audio_path)

    all_words = []

    try:
        for chunk_path, offset in chunk_list:
            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

            chunk_words = getattr(transcription, "words", None)

            if chunk_words:
                for w in chunk_words:
                    all_words.append(
                        {"text": w["word"].strip(), "start": w["start"] + offset, "end": w["end"] + offset}
                    )
            else:
                for seg in transcription.segments:
                    all_words.append(
                        {"text": seg["text"].strip(), "start": seg["start"] + offset, "end": seg["end"] + offset}
                    )
    finally:
        if chunk_dir:
            for chunk_path, _ in chunk_list:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            if os.path.isdir(chunk_dir) and not os.listdir(chunk_dir):
                os.rmdir(chunk_dir)

    # Drop any word that touches a FunASR-confirmed silence zone at all
    # -- full overlap, partial overlap, or just sharing an endpoint.
    before = len(all_words)
    all_words = [w for w in all_words if not _overlaps_silence(w, silence_zones)]
    dropped = before - len(all_words)
    if dropped:
        print(f"Dropped {dropped} word(s) touching FunASR-detected silence.")

    sentences = _words_to_sentences(all_words, silence_zones)
    print(f"Rebuilt {len(sentences)} sentence(s) from surviving words.")

    segments = _sentences_to_segments(
        sentences,
        min_segment_seconds=min_segment_seconds,
        max_gap_seconds=max_segment_gap_seconds,
    )
    print(
        f"Merged into {len(segments)} final segment(s) "
        f"(target >= {min_segment_seconds:.0f}s, early-close gap >= {max_segment_gap_seconds:.0f}s)."
    )

    # Entire transcript as one block of text, in addition to the
    # per-segment breakdown returned above.
    full_text = " ".join(s["text"].strip() for s in segments).strip()

    if not keep_audio and os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "segments": segments,
        "text": full_text,
        "max_duration": round(total_duration, 2),
        "silences": silence_zones,
    }


# if __name__ == "__main__":
#     V = input("Enter URL : ").strip()
#     transcript = ingestion(video_url=V)
#     print(f"{len(transcript['segments'])} segments, "
#           f"{len(transcript['text'])} chars of full text, "
#           f"{len(transcript['silences'])} silence range(s).")
