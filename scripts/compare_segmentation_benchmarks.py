"""Compare two real segmentation benchmark output directories."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_benchmarks(left_dir, right_dir, left_label, right_label, output_dir):
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    left_results = pd.read_csv(left_dir / "results.csv")
    right_results = pd.read_csv(right_dir / "results.csv")
    left_tracks = pd.read_csv(left_dir / "tracks.csv")
    right_tracks = pd.read_csv(right_dir / "tracks.csv")

    track_cmp = left_tracks[
        ["track", "cross_smooth_points", "cross_boundary_points", "intra_points", "transition_edges"]
    ].merge(
        right_tracks[
            ["track", "cross_smooth_points", "cross_boundary_points", "intra_points", "transition_edges"]
        ],
        on="track",
        suffixes=(f"_{left_label}", f"_{right_label}"),
    )
    result_cmp = left_results[
        ["track", "target_ratio", "duration_error_seconds", "path_runtime_ms", "n_jumps"]
    ].merge(
        right_results[
            ["track", "target_ratio", "duration_error_seconds", "path_runtime_ms", "n_jumps"]
        ],
        on=["track", "target_ratio"],
        suffixes=(f"_{left_label}", f"_{right_label}"),
    )

    track_cmp.to_csv(output_dir / f"{left_label}_vs_{right_label}_tracks.csv", index=False)
    result_cmp.to_csv(output_dir / f"{left_label}_vs_{right_label}_results.csv", index=False)

    labels = track_cmp["track"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - width / 2, track_cmp[f"cross_smooth_points_{left_label}"], width, label=left_label)
    ax.bar(x + width / 2, track_cmp[f"cross_smooth_points_{right_label}"], width, label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("smooth cross points")
    ax.set_title("Smooth cross-segment candidates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{left_label}_vs_{right_label}_smooth_cross_points.png", dpi=150)
    plt.close(fig)

    case_labels = [f"{row.track}\n{row.target_ratio}" for row in result_cmp.itertuples()]
    x = np.arange(len(case_labels))
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.bar(x - width / 2, result_cmp[f"duration_error_seconds_{left_label}"], width, label=left_label)
    ax.bar(x + width / 2, result_cmp[f"duration_error_seconds_{right_label}"], width, label=right_label)
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("absolute duration error (s)")
    ax.set_title("Duration error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{left_label}_vs_{right_label}_duration_error.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_benchmarks(
        args.left_dir,
        args.right_dir,
        args.left_label,
        args.right_label,
        args.output_dir,
    )
