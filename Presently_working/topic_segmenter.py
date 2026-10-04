"""
Steps 5-11 of the pipeline: turn per-segment embeddings into
candidate topic boundaries, then into topic segments.

    segment embeddings
            |
    left/right contextual windows   (compute_boundary_scores)
            |
    cosine similarity -> boundary strength
            |
    smoothing                        (smooth_scores)
            |
    local maxima / peak detection    (detect_candidate_boundaries)
            |
    topic segments                   (build_topics)
"""
import numpy as np
from scipy.signal import find_peaks

import config


def _window_embedding(embeddings: np.ndarray, indices: range, weights: np.ndarray | None) -> np.ndarray:
    """
    Reduces a set of segment embeddings (identified by `indices` into
    the full `embeddings` array) to a single vector representing that
    window, then re-normalizes it.

    Plain mean by default (config.ENABLE_SHORT_SEGMENT_DOWNWEIGHTING =
    False). When enabled, `weights` down-weights short/filler segments
    so a single one-word segment ("Okay.") can't dominate a 10-segment
    window's centroid.
    """
    idx = list(indices)
    vectors = embeddings[idx]

    if weights is None:
        centroid = vectors.mean(axis=0)
    else:
        w = weights[idx]
        centroid = np.average(vectors, axis=0, weights=w)

    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _segment_weights(segments: list[dict]) -> np.ndarray | None:
    if not config.ENABLE_SHORT_SEGMENT_DOWNWEIGHTING:
        return None
    weights = np.ones(len(segments), dtype=np.float32)
    for i, seg in enumerate(segments):
        word_count = len(seg["text"].split())
        if word_count < config.SHORT_SEGMENT_WORD_THRESHOLD:
            weights[i] = config.SHORT_SEGMENT_WEIGHT
    return weights


def compute_boundary_scores(
    segments: list[dict],
    embeddings: np.ndarray,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> list[dict]:
    """
    For every valid boundary position (a full, non-overlapping
    window_size-segment window on both sides), computes:

        left_embedding  = normalized mean of embeddings[i-window_size : i]
        right_embedding = normalized mean of embeddings[i : i+window_size]
        similarity      = cosine_similarity(left_embedding, right_embedding)
        boundary_score  = 1 - similarity

    A boundary "after segment i-1" sits between segment i-1 and
    segment i. Edge handling (step 8): boundaries are only evaluated
    where BOTH windows are completely full (no partial windows), i.e.
    for i in [window_size, N - window_size]. This keeps every boundary
    score computed from an equal amount of context, so scores are
    directly comparable across the whole transcript -- the first
    (window_size - 1) and last (window_size - 1) possible boundary
    positions near the very start/end of the transcript are simply not
    scored at all, and always fall inside the first/last topic.

    Returns a list of dicts (ordered by position), one per evaluated
    boundary, with keys: boundary_after_segment, left_start, left_end,
    right_start, right_end, similarity, boundary_score. Positions
    (left_start/left_end/right_start/right_end/boundary_after_segment)
    are transcript segment IDs (segments[i]["id"]), not raw array
    indices, so they read correctly even if segment ids ever have
    gaps.
    """
    n = len(segments)
    weights = _segment_weights(segments)
    results = []

    if n < 2 * window_size:
        print(
            f"Warning: only {n} segments but window_size={window_size} needs at "
            f"least {2 * window_size} to score any boundary. No boundaries scored."
        )
        return results

    for i in range(window_size, n - window_size + 1, stride):
        left_range = range(i - window_size, i)
        right_range = range(i, i + window_size)

        left_vec = _window_embedding(embeddings, left_range, weights)
        right_vec = _window_embedding(embeddings, right_range, weights)

        similarity = float(np.dot(left_vec, right_vec))
        similarity = max(-1.0, min(1.0, similarity))  # guard fp drift
        boundary_score = 1.0 - similarity

        results.append(
            {
                "boundary_after_segment": segments[i - 1]["id"],
                "left_start": segments[i - window_size]["id"],
                "left_end": segments[i - 1]["id"],
                "right_start": segments[i]["id"],
                "right_end": segments[min(i + window_size - 1, n - 1)]["id"],
                "similarity": similarity,
                "boundary_score": boundary_score,
            }
        )

    return results


def smooth_scores(
    boundary_results: list[dict],
    smoothing_window: int = config.SMOOTHING_WINDOW,
) -> list[dict]:
    """
    Adds a "smoothed_boundary_score" key to each entry in
    boundary_results, computed as a simple centered moving average
    over the raw "boundary_score" values with the given window width.
    smoothing_window = 1 is a no-op (smoothed == raw).

    Mutates and returns the same list (in place) for convenience.
    """
    if not boundary_results:
        return boundary_results

    raw = np.array([r["boundary_score"] for r in boundary_results], dtype=np.float64)

    if smoothing_window <= 1:
        smoothed = raw
    else:
        kernel = np.ones(smoothing_window) / smoothing_window
        # 'same' mode + edge padding via 'reflect' keeps the array the
        # same length and avoids artificially dragging the first/last
        # few scores toward zero the way naive zero-padding would.
        pad = smoothing_window // 2
        padded = np.pad(raw, (pad, pad), mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")[: len(raw)]

    for r, s in zip(boundary_results, smoothed):
        r["smoothed_boundary_score"] = float(s)

    return boundary_results


def detect_candidate_boundaries(
    boundary_results: list[dict],
    min_boundary_distance: int = config.MIN_BOUNDARY_DISTANCE,
    min_topic_length: int = config.MIN_TOPIC_LENGTH,
    prominence: float | None = config.BOUNDARY_PROMINENCE,
    prominence_std_multiplier: float = config.BOUNDARY_PROMINENCE_STD_MULTIPLIER,
) -> list[dict]:
    """
    Runs scipy.signal.find_peaks on the smoothed boundary-score curve
    to find local maxima (== local minima in left/right similarity),
    i.e. candidate topic boundaries. Adaptive rather than a fixed
    threshold, because similarity distributions vary a lot between
    recordings (a lecture with lots of topic changes will have a
    different score distribution than a single-topic Q&A session).

    - distance=min_boundary_distance stops two noisy adjacent peaks
      from both being reported (since STRIDE=1, this is directly in
      units of transcript segments).
    - prominence: if None (the default), auto-derived per-transcript
      as prominence_std_multiplier * std(smoothed_scores), so the
      threshold scales with how noisy/spread-out THIS transcript's
      scores are, rather than assuming one fixed cutoff works for
      every lecture.
    - After scipy's peaks come back, a second pass enforces
      min_topic_length as a hard floor: any accepted boundary that
      would leave the run touching the transcript ends. Full topic
      length enforcement (between resulting topics) happens later in
      build_topics/merge; here we simply filter unreasonably close
      peaks that slipped through if min_boundary_distance was set
      looser than min_topic_length.

    Mutates boundary_results in place, setting is_candidate=True/False
    on every entry, and also returns the sorted list of candidate
    boundary_after_segment values.
    """
    for r in boundary_results:
        r["is_candidate"] = False

    if not boundary_results:
        return []

    smoothed = np.array([r["smoothed_boundary_score"] for r in boundary_results])

    if prominence is None:
        std = float(np.std(smoothed))
        prominence = max(std * prominence_std_multiplier, 1e-6)

    peak_indices, _ = find_peaks(
        smoothed,
        distance=max(1, min_boundary_distance),
        prominence=prominence,
    )

    # Second filter: enforce min_topic_length between consecutive
    # accepted peaks too (in case min_boundary_distance was configured
    # smaller than min_topic_length during experimentation).
    accepted_positions = []
    last_pos = -min_topic_length  # so the very first peak is never rejected here
    for pos in peak_indices:
        if pos - last_pos >= min_topic_length:
            accepted_positions.append(pos)
            last_pos = pos

    for pos in accepted_positions:
        boundary_results[pos]["is_candidate"] = True

    candidate_boundaries = [boundary_results[pos]["boundary_after_segment"] for pos in accepted_positions]
    return candidate_boundaries


def build_topics(segments: list[dict], candidate_boundaries: list[int], boundary_results: list[dict]) -> list[dict]:
    """
    Converts a list of candidate boundary_after_segment values into
    contiguous topic segments covering every original segment exactly
    once (step 11/12).

    boundary_score assigned to a topic represents the strength of the
    boundary at the END of that topic, using the SMOOTHED score (the
    same curve peak detection ran on) at that boundary position. The
    final topic has no following boundary, so its boundary_score is
    null.
    """
    id_to_score = {
        r["boundary_after_segment"]: r["smoothed_boundary_score"]
        for r in boundary_results
        if r["is_candidate"]
    }

    boundaries_sorted = sorted(candidate_boundaries)

    id_to_index = {seg["id"]: idx for idx, seg in enumerate(segments)}

    topics = []
    start_idx = 0
    topic_id = 1

    for boundary_id in boundaries_sorted:
        end_idx = id_to_index[boundary_id]
        topic_segments = segments[start_idx : end_idx + 1]
        if not topic_segments:
            continue
        topics.append(
            {
                "topic_id": topic_id,
                "start_segment": topic_segments[0]["id"],
                "end_segment": topic_segments[-1]["id"],
                "start_time": topic_segments[0]["start"],
                "end_time": topic_segments[-1]["end"],
                "segments": topic_segments,
                "boundary_score": id_to_score.get(boundary_id),
            }
        )
        topic_id += 1
        start_idx = end_idx + 1

    # Final topic: everything after the last boundary, no following
    # boundary -> boundary_score = null.
    if start_idx < len(segments):
        topic_segments = segments[start_idx:]
        topics.append(
            {
                "topic_id": topic_id,
                "start_segment": topic_segments[0]["id"],
                "end_segment": topic_segments[-1]["id"],
                "start_time": topic_segments[0]["start"],
                "end_time": topic_segments[-1]["end"],
                "segments": topic_segments,
                "boundary_score": None,
            }
        )

    return topics
