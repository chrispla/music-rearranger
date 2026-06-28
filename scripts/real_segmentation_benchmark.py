"""Run the real segmentation/cutpoint/rearrangement pipeline on MP3s."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import pickle
import sys
import time

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rearranger.construction import construct_audio  # noqa: E402
from rearranger.formatting import (  # noqa: E402
    get_target_n_beats,
    get_unique_segments,
    quantize_to_measures,
    structure_time_to_beats,
)
from rearranger.identification import (  # noqa: E402
    common_patterns,
    cross_segment_points,
    intra_segment_points,
)
from rearranger.optimization import (  # noqa: E402
    get_transitions,
    sparse_dijkstra_search,
    sparse_dijkstra_seconds_search,
)
from rearranger.plotting import save_useful_plots  # noqa: E402
from rearranger.segmentation import fast_segmentation, precise_segmentation  # noqa: E402
from scripts.real_audio_benchmark import (  # noqa: E402
    TRACKS,
    download_track,
    safe_name,
)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def path_duration(path):
    return sum(1 for current, target in zip(path[:-1], path[1:]) if target == current + 1)


def path_jumps(path):
    return [
        (int(current), int(target))
        for current, target in zip(path[:-1], path[1:])
        if target != current + 1
    ]


def path_intervals(path):
    intervals = []
    start = int(path[0])
    previous = int(path[0])
    for beat in path[1:]:
        beat = int(beat)
        if beat == previous + 1:
            previous = beat
            continue
        intervals.append((start, previous))
        start = beat
        previous = beat
    intervals.append((start, previous))
    return [(start, end) for start, end in intervals if end > start]


def plot_path_timeline(path, total_beats, title, out_path):
    jumps = path_jumps(path)
    intervals = path_intervals(path)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.hlines(0, 0, total_beats, color="#bbbbbb", linewidth=2)
    for start, end in intervals:
        ax.hlines(0, start, end, color="#2f6f73", linewidth=7, alpha=0.75)
    for idx, (start, end) in enumerate(jumps):
        height = 0.35 + (idx % 4) * 0.14
        if end < start:
            height *= -1
        color = "#d08b38" if end > start else "#6457a6"
        ax.annotate(
            "",
            xy=(end, 0),
            xytext=(start, 0),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "connectionstyle": f"arc3,rad={height}",
                "linewidth": 1.5,
            },
        )
    ax.set_title(title)
    ax.set_xlabel("source beat")
    ax.set_yticks([])
    ax.set_xlim(0, total_beats)
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transition_diagnostics(patterns, transition_points, selected_jumps, title, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].imshow(patterns, origin="lower", aspect="auto", cmap="magma")
    if transition_points:
        xs = [point[0][0] for point in transition_points]
        ys = [point[0][1] for point in transition_points]
        axes[0].scatter(xs, ys, s=8, facecolors="none", edgecolors="#64c7a4", linewidths=0.5)
    if selected_jumps:
        xs = [jump[0] for jump in selected_jumps]
        ys = [jump[1] for jump in selected_jumps]
        axes[0].scatter(xs, ys, s=28, marker="x", color="#f7f06d", linewidths=1.2)
    axes[0].set_title("patterns and transitions")
    axes[0].set_xlabel("exit beat")
    axes[0].set_ylabel("entry beat")

    similarities = [point[1] for point in transition_points if point[1] != 0]
    if similarities:
        axes[1].hist(similarities, bins=20, color="#2f6f73", alpha=0.8)
    axes[1].set_title("smooth transition similarities")
    axes[1].set_xlabel("similarity")
    axes[1].set_ylabel("count")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(rows, track_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = [row["track"] for row in track_rows]
    smooth = [row["cross_smooth_points"] for row in track_rows]
    boundary = [row["cross_boundary_points"] for row in track_rows]
    intra = [row["intra_points"] for row in track_rows]
    x = np.arange(len(labels))
    ax.bar(x, boundary, label="cross boundary fallback", color="#c9c0aa")
    ax.bar(x, smooth, bottom=boundary, label="cross smooth", color="#2f6f73")
    ax.bar(x, intra, bottom=np.asarray(boundary) + np.asarray(smooth), label="intra", color="#6457a6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("candidate points")
    ax.set_title("Candidate transition points from real segmentation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_points.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    case_labels = [f"{row['track']} {row['target_ratio']}" for row in rows]
    runtimes = [row["path_runtime_ms"] for row in rows]
    ax.bar(np.arange(len(rows)), runtimes, color="#2f6f73")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(case_labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("ms")
    ax.set_title("Sparse path-search runtime")
    fig.tight_layout()
    fig.savefig(out_dir / "path_runtime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    errors = [row["duration_error_seconds"] for row in rows]
    ax.bar(np.arange(len(rows)), errors, color="#d08b38")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(case_labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("Output duration error")
    fig.tight_layout()
    fig.savefig(out_dir / "duration_error.png", dpi=150)
    plt.close(fig)


def write_dashboard(out_path, rows, track_rows):
    result_lines = []
    for row in rows:
        result_lines.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{row['target_ratio']}</td>"
            f"<td>{row['target_seconds']}</td>"
            f"<td>{row['output_seconds']:.2f}</td>"
            f"<td>{row['duration_error_seconds']:.2f}</td>"
            f"<td>{row['n_jumps']}</td>"
            f"<td>{row['mean_jump_similarity']:.3f}</td>"
            f"<td>{row['path_runtime_ms']:.3f}</td>"
            f"<td>{html.escape(row['audio_output'])}</td>"
            "</tr>"
        )
    track_json = html.escape(json.dumps(track_rows, indent=2))
    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Real segmentation benchmark</title>"
        "<style>body{font-family:Arial,sans-serif;margin:28px;max-width:1200px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left}"
        "th{background:#f0f0ec} img{max-width:100%;height:auto}</style></head><body>"
        "<h1>Real segmentation benchmark</h1>"
        "<p>This benchmark uses the repo's real segmentation, transition "
        "identification, sparse path search, and audio reconstruction.</p>"
        "<h2>Summary Plots</h2>"
        "<p><img src='plots/candidate_points.png'></p>"
        "<p><img src='plots/path_runtime.png'></p>"
        "<p><img src='plots/duration_error.png'></p>"
        "<h2>Cases</h2>"
        "<table><tr><th>Track</th><th>Target ratio</th><th>Target s</th>"
        "<th>Output s</th><th>Error s</th><th>Jumps</th>"
        "<th>Mean jump sim</th><th>Path ms</th><th>Audio</th></tr>"
        + "\n".join(result_lines)
        + "</table><h2>Track Diagnostics</h2><pre>"
        + track_json
        + "</pre></body></html>"
    )


def run_track(title, filename, config, args, audio_dir, out_dir):
    slug = safe_name(title)
    audio_path = audio_dir / f"{slug}.mp3"
    download_track(filename, audio_path)

    track_dir = out_dir / slug
    track_dir.mkdir(parents=True, exist_ok=True)
    cache_track_dir = Path(args.segmentation_cache_dir) / slug
    cache_track_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = cache_track_dir / f"{slug}_{args.seg_method}.pkl"

    if pkl_path.exists() and not args.force:
        with pkl_path.open("rb") as handle:
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = pickle.load(handle)
    else:
        if args.seg_method == "fast":
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = fast_segmentation(
                audio_filepath=str(audio_path),
                config=config,
            )
        else:
            segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync = precise_segmentation(
                audio_filepath=str(audio_path),
                config=config,
                deepsim_model_dir=args.deepsim_model_dir,
                fewshot_model_dir=args.fewshot_model_dir,
                use_gpu=args.use_gpu,
            )
        with pkl_path.open("wb") as handle:
            pickle.dump([segmentation, beat_times, beat_analysis, R, Csync, Msync, Hsync], handle)

    if args.plot_segmentation:
        save_useful_plots(
            output_dir=str(track_dir),
            output_name=slug,
            seg_method=args.seg_method,
            segmentation=segmentation,
            Csync=Csync,
            Msync=Msync,
            Hsync=Hsync,
        )

    segmentation_beats = structure_time_to_beats(segmentation, beat_times)
    segmentation_n_measures, _, _, _ = quantize_to_measures(
        segmentation_beats=segmentation_beats,
        n_measures=config["min_measure"],
        beat_analysis=beat_analysis,
        beat_times=beat_times,
    )
    segmentation_n_measures_unique = get_unique_segments(segmentation_n_measures)
    beats_in_measure = int(np.max(beat_analysis[:, 1]))

    patterns = common_patterns(
        Csync=Csync,
        Msync=Msync,
        Hsync=Hsync,
        length=config["pattern_length"],
        percentile=config["similarity_percentile"],
    )
    cross_points = cross_segment_points(
        segmentation=segmentation_n_measures_unique,
        quantization=config["min_measure"],
        beats_in_measure=beats_in_measure,
        patterns=patterns,
    )
    intra_points = intra_segment_points(
        segmentation=segmentation_n_measures,
        levels_list=config["intra_levels_list"],
        min_d_len=config["pattern_length"],
        patterns=patterns,
        beats_in_measure=beats_in_measure,
    )
    transitions, similarities = get_transitions(
        points=cross_points + intra_points,
        n_beats=patterns.shape[0],
        type=args.transition_types or config["transition_types"],
        neighbors=True,
    )

    y, sr = librosa.load(path=str(audio_path), sr=None)
    original_seconds = len(y) / sr
    track_row = {
        "seg_method": args.seg_method,
        "track": title,
        "audio": str(audio_path),
        "original_seconds": round(original_seconds, 3),
        "beats": len(beat_times),
        "beats_in_measure": beats_in_measure,
        "cross_points": len(cross_points),
        "cross_smooth_points": len([point for point in cross_points if point[1] != 0]),
        "cross_boundary_points": len([point for point in cross_points if point[1] == 0]),
        "intra_points": len(intra_points),
        "transition_edges": sum(max(0, len(values) - 1) for values in transitions.values()),
    }

    rows = []
    for ratio in args.target_ratios:
        target_seconds = int(round(original_seconds * ratio))
        target_beats = get_target_n_beats(target_seconds, beat_analysis)
        start_time = time.perf_counter()
        if args.search_objective == "seconds":
            beat_list = sparse_dijkstra_seconds_search(
                transitions=transitions,
                similarities=similarities,
                target_seconds=target_seconds,
                total_beats=patterns.shape[0],
                beat_times=beat_times,
                beat_analysis=beat_analysis,
                seconds_quantization=args.seconds_quantization,
            )
        else:
            beat_list = sparse_dijkstra_search(
                transitions=transitions,
                similarities=similarities,
                target_beats=target_beats,
                total_beats=patterns.shape[0],
                beat_analysis=beat_analysis,
                tolerance_beats=config.get("path_tolerance_beats"),
                max_tolerance_beats=config.get("path_max_tolerance_beats"),
            )
        path_runtime_ms = (time.perf_counter() - start_time) * 1000

        audio_out = track_dir / f"{slug}_{target_seconds}s.wav"
        y_rearranged = construct_audio(
            y=y,
            sr=sr,
            recon_beats=beat_list,
            beat_times=beat_times,
            crossfade=config["crossfade"],
        )
        sf.write(audio_out, y_rearranged, sr)

        jumps = path_jumps(beat_list)
        jump_similarities = [
            similarities.get(jump, 0)
            for jump in jumps
        ]
        row = {
            "seg_method": args.seg_method,
            "track": title,
            "target_ratio": ratio,
            "target_seconds": target_seconds,
            "target_beats": target_beats,
            "output_beats": path_duration(beat_list),
            "output_seconds": len(y_rearranged) / sr,
            "duration_error_seconds": abs((len(y_rearranged) / sr) - target_seconds),
            "n_jumps": len(jumps),
            "mean_jump_similarity": float(np.mean(jump_similarities)) if jump_similarities else 1.0,
            "min_jump_similarity": float(np.min(jump_similarities)) if jump_similarities else 1.0,
            "path_runtime_ms": path_runtime_ms,
            "audio_output": str(audio_out.relative_to(out_dir)),
            "timeline_plot": str((track_dir / f"{slug}_{target_seconds}s_timeline.png").relative_to(out_dir)),
            "transition_plot": str((track_dir / f"{slug}_transitions.png").relative_to(out_dir)),
            "jumps": json.dumps(jumps),
        }
        plot_path_timeline(
            beat_list,
            patterns.shape[0],
            f"{title}: {target_seconds}s target",
            track_dir / f"{slug}_{target_seconds}s_timeline.png",
        )
        rows.append(row)

    selected_jumps = []
    for row in rows:
        selected_jumps.extend(json.loads(row["jumps"]))
    plot_transition_diagnostics(
        patterns,
        cross_points + intra_points,
        selected_jumps,
        title,
        track_dir / f"{slug}_transitions.png",
    )

    return track_row, rows


def run_benchmark(args):
    out_dir = Path(args.output_dir)
    audio_dir = Path(args.audio_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    all_rows = []
    track_rows = []
    for title, filename in TRACKS[:args.limit]:
        print(f"> {title} ({args.seg_method})")
        track_row, rows = run_track(title, filename, config, args, audio_dir, out_dir)
        track_rows.append(track_row)
        all_rows.extend(rows)
        print(
            f"  transitions: cross smooth={track_row['cross_smooth_points']}, "
            f"boundary={track_row['cross_boundary_points']}, intra={track_row['intra_points']}"
        )
        for row in rows:
            print(
                f"  target {row['target_ratio']}: {row['output_seconds']:.2f}s, "
                f"{row['n_jumps']} jumps, {row['path_runtime_ms']:.2f} ms"
            )

    write_csv(
        out_dir / "results.csv",
        all_rows,
        [
            "track",
            "seg_method",
            "target_ratio",
            "target_seconds",
            "target_beats",
            "output_beats",
            "output_seconds",
            "duration_error_seconds",
            "n_jumps",
            "mean_jump_similarity",
            "min_jump_similarity",
            "path_runtime_ms",
            "audio_output",
            "timeline_plot",
            "transition_plot",
            "jumps",
        ],
    )
    write_csv(
        out_dir / "tracks.csv",
        track_rows,
        [
            "track",
            "seg_method",
            "audio",
            "original_seconds",
            "beats",
            "beats_in_measure",
            "cross_points",
            "cross_smooth_points",
            "cross_boundary_points",
            "intra_points",
            "transition_edges",
        ],
    )
    plot_summary(all_rows, track_rows, out_dir / "plots")
    write_dashboard(out_dir / "index.html", all_rows, track_rows)
    print(f"> Wrote {out_dir / 'index.html'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="output/real_audio_benchmark/audio")
    parser.add_argument("--output-dir", default="output/real_segmentation_benchmark")
    parser.add_argument("--segmentation-cache-dir", default=None)
    parser.add_argument("--seg-method", choices=["fast", "precise"], default="fast")
    parser.add_argument("--deepsim-model-dir", default="models/deepsim")
    parser.add_argument("--fewshot-model-dir", default="models/fewshot")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--target-ratios", type=float, nargs="+", default=[0.65, 0.8, 1.2])
    parser.add_argument("--transition-types", choices=["ideal", "nonnegative", "all"])
    parser.add_argument("--search-objective", choices=["beats", "seconds"], default="beats")
    parser.add_argument("--seconds-quantization", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-segmentation", action="store_true")
    args = parser.parse_args()
    if args.segmentation_cache_dir is None:
        args.segmentation_cache_dir = args.output_dir
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
