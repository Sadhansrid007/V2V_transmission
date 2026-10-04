"""
Entry point. Run with:

    python run_segmentation.py "https://example.com/lecture.mp4"

or set VIDEO_URL below and just run:

    python run_segmentation.py

video_url is required: it's used to key transcript.json (which video's
segments to load), to select this video's own ChromaDB collection, and to
namespace every output this script writes (topic_segmentation.json,
boundary_scores.json, and boundary_scores.png), so running this on
multiple videos in the same project directory never overwrites another
video's results. See config.url_key / config.collection_name_for_url /
config.url_scoped_path.

Performs the full pipeline described in the spec:
  1. Load transcript.json
  2/3. Generate (or reuse) segment embeddings, store in ChromaDB
  4. Calculate sliding left/right window similarities
  5. Calculate boundary scores
  6. Smooth scores
  7. Detect candidate boundaries
  8. Generate topic_segmentation.json
  9. Generate boundary_scores.json
  10. Generate boundary_scores.png
  11. Print a summary
"""
import sys

import config
from segment_embeddings import load_transcript, get_or_compute_embeddings
from topic_segmenter import (
    compute_boundary_scores,
    smooth_scores,
    detect_candidate_boundaries,
    build_topics,
)

# Leave blank to be prompted, or pass the URL as a command-line argument.
VIDEO_URL = ""


def save_topic_segmentation(topics: list[dict], video_url: str, path: str = config.OUTPUT_TOPIC_JSON) -> None:
    config.save_url_keyed_entry(path, video_url, topics)


def save_boundary_scores(boundary_results: list[dict], video_url: str, path: str = config.OUTPUT_BOUNDARY_JSON) -> None:
    config.save_url_keyed_entry(path, video_url, boundary_results)


def plot_boundary_scores(boundary_results: list[dict], video_url: str, path: str = config.OUTPUT_PLOT_PNG) -> str:
    """Saves to a per-video filename (config.url_scoped_path) since a PNG
    can't live inside a shared url-keyed JSON dict the way the other
    outputs do. Returns the actual path written to."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = config.url_scoped_path(path, video_url)

    if not boundary_results:
        print("No boundary scores to plot (transcript too short for window_size).")
        return out_path

    x = [r["boundary_after_segment"] for r in boundary_results]
    raw = [r["boundary_score"] for r in boundary_results]
    smoothed = [r["smoothed_boundary_score"] for r in boundary_results]
    candidate_x = [r["boundary_after_segment"] for r in boundary_results if r["is_candidate"]]
    candidate_y = [r["smoothed_boundary_score"] for r in boundary_results if r["is_candidate"]]

    plt.figure(figsize=(14, 6))
    plt.plot(x, raw, color="lightgray", linewidth=1, label="raw boundary score")
    plt.plot(x, smoothed, color="steelblue", linewidth=2, label="smoothed boundary score")
    plt.scatter(candidate_x, candidate_y, color="crimson", zorder=5, s=50, label="candidate boundary")

    plt.xlabel("Transcript segment number")
    plt.ylabel("Boundary strength (1 - cosine similarity)")
    plt.title("Semantic boundary scores across transcript")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _resolve_video_url(video_url: str | None) -> str:
    if video_url:
        return video_url
    video_url = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or VIDEO_URL.strip()
    if not video_url:
        video_url = input("Enter video URL (used to key transcript.json / chroma_db / outputs): ").strip()
    if not video_url:
        print("ERROR: no video URL provided.")
        sys.exit(1)
    return video_url


def main(video_url: str | None = None):
    video_url = _resolve_video_url(video_url)
    print(f"video_url = {video_url}")
    print(f"url key   = {config.url_key(video_url)}\n")

    segments = load_transcript(video_url, config.TRANSCRIPT_PATH)
    print(f"Loaded {len(segments)} transcript segments\n")

    embeddings = get_or_compute_embeddings(segments, video_url)
    print(f"Embedding model: {config.EMBEDDING_MODEL}")
    print(f"Embedding dimension: {embeddings.shape[1]}\n")

    print(f"Window size: {config.WINDOW_SIZE}")
    print(f"Stride: {config.STRIDE}\n")

    boundary_results = compute_boundary_scores(segments, embeddings)
    print(f"Calculated {len(boundary_results)} boundary scores\n")

    smooth_scores(boundary_results)
    candidate_boundaries = detect_candidate_boundaries(boundary_results)
    print(f"Detected {len(candidate_boundaries)} candidate topic boundaries\n")

    topics = build_topics(segments, candidate_boundaries, boundary_results)
    print(f"Generated {len(topics)} topics\n")

    save_topic_segmentation(topics, video_url)
    save_boundary_scores(boundary_results, video_url)
    plot_path = plot_boundary_scores(boundary_results, video_url)

    for t in topics:
        print(f"Topic {t['topic_id']}: segments {t['start_segment']}-{t['end_segment']} "
              f"| {t['start_time']:.2f}s-{t['end_time']:.2f}s")

    print("\nSaved:")
    print(f"  {config.OUTPUT_TOPIC_JSON}       (entry for key {config.url_key(video_url)})")
    print(f"  {config.OUTPUT_BOUNDARY_JSON}    (entry for key {config.url_key(video_url)})")
    print(f"  {plot_path}")
    print(f"  {config.CHROMA_PATH}/  (collection '{config.collection_name_for_url(video_url)}')")


if __name__ == "__main__":
    main()
