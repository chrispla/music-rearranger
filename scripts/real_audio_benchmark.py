"""Benchmark sparse path search on a small open-licensed MP3 corpus.

This script intentionally avoids the heavyweight segmentation dependencies so
the path finder can be tested in a plain Python environment with ffmpeg and
numpy. It estimates a rough beat grid, builds audio-derived transition
candidates from local spectral similarity, runs the sparse Dijkstra search, and
writes CSV, SVG plots, an HTML dashboard, and optional preview MP3s.
"""

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rearranger.optimization import sparse_dijkstra_search  # noqa: E402


COMMONS_REDIRECT = "https://commons.wikimedia.org/wiki/Special:Redirect/file/"
COMMONS_FILE_PAGE = "https://commons.wikimedia.org/wiki/File:"

TRACKS = [
    ("Agnus Dei X", "Agnus Dei X (ISRC USUAN1100679).mp3"),
    ("Amazing Grace 2011", "Amazing Grace 2011 (ISRC USUAN1100820).mp3"),
    ("Blue Feather", "Blue Feather (ISRC USUAN1100524).mp3"),
    ("Consort for Brass", "Consort for Brass (ISRC USUAN1100681).mp3"),
    ("Exotic Battle", "Exotic Battle (ISRC USUAN1100451).mp3"),
    ("Funeral March for Brass", "Funeral March for Brass (ISRC USUAN1100797).mp3"),
    ("Ghost Dance", "Ghost Dance (ISRC USUAN1100573).mp3"),
    ("Intended Force", "Intended Force (ISRC USUAN1100195).mp3"),
    ("Meatball Parade", "Meatball Parade (ISRC USUAN1500014).mp3"),
    ("Minstrel Guild", "Minstrel Guild (ISRC USUAN1200098).mp3"),
]


def safe_name(name):
    keep = []
    for char in name.lower():
        if char.isalnum():
            keep.append(char)
        elif keep and keep[-1] != "_":
            keep.append("_")
    return "".join(keep).strip("_")


def commons_download_url(filename):
    return COMMONS_REDIRECT + quote(filename)


def commons_page_url(filename):
    return COMMONS_FILE_PAGE + quote(filename)


def download_track(filename, out_path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    url = commons_download_url(filename)
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--user-agent",
        "music-rearranger-real-audio-benchmark/1.0",
        "--output",
        str(out_path),
        url,
    ]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        request = Request(
            url,
            headers={"User-Agent": "music-rearranger-real-audio-benchmark/1.0"},
        )
        with urlopen(request, timeout=120) as response:
            data = response.read()
        out_path.write_bytes(data)


def run_command(command, input_bytes=None):
    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def decode_audio(path, sample_rate, max_seconds):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = run_command(command)
    return np.frombuffer(result.stdout, dtype=np.float32)


def encode_mp3(samples, sample_rate, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "-",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(out_path),
    ]
    run_command(command, samples.astype(np.float32).tobytes())


def frame_rms(samples, frame_size=1024, hop_size=512):
    n_frames = 1 + max(0, (len(samples) - frame_size) // hop_size)
    rms = np.zeros(n_frames, dtype=np.float32)
    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        frame = samples[start:start + frame_size]
        rms[frame_idx] = math.sqrt(float(np.mean(frame * frame)) + 1e-12)
    return rms


def estimate_beat_period(samples, sample_rate):
    hop_size = 512
    rms = frame_rms(samples, hop_size=hop_size)
    onset = np.maximum(0, np.diff(rms, prepend=rms[0]))
    if np.max(onset) > 0:
        onset = onset / np.max(onset)

    corr = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    frame_seconds = hop_size / sample_rate
    min_lag = max(1, int(round((60 / 180) / frame_seconds)))
    max_lag = min(len(corr) - 1, int(round((60 / 60) / frame_seconds)))
    if max_lag <= min_lag:
        return 0.5, 120.0

    lag_range = np.arange(min_lag, max_lag + 1)
    # Gentle preference for common musical tempi when autocorrelation is flat.
    bpm = 60 / (lag_range * frame_seconds)
    tempo_prior = np.exp(-((bpm - 115) ** 2) / (2 * 35 ** 2))
    best_lag = lag_range[int(np.argmax(corr[lag_range] * (0.75 + 0.25 * tempo_prior)))]
    beat_period = best_lag * frame_seconds
    return beat_period, 60 / beat_period


def beat_boundaries(samples, sample_rate, beat_period):
    beat_samples = max(64, int(round(beat_period * sample_rate)))
    boundaries = list(range(0, len(samples), beat_samples))
    if boundaries[-1] != len(samples):
        boundaries.append(len(samples))
    return np.asarray(boundaries, dtype=np.int64)


def spectral_feature(segment, sample_rate, n_bands=28):
    if len(segment) < 64:
        return np.zeros(n_bands + 1, dtype=np.float32)

    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(len(segment), 1 / sample_rate)
    edges = np.geomspace(60, sample_rate / 2, n_bands + 1)
    values = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        values.append(float(np.mean(spectrum[mask])) if np.any(mask) else 0)
    values.append(float(np.sqrt(np.mean(segment * segment)) + 1e-12))
    feature = np.log1p(np.asarray(values, dtype=np.float32))
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature = feature / norm
    return feature


def boundary_features(samples, sample_rate, boundaries):
    beat_span = int(np.median(np.diff(boundaries)))
    features = []
    for boundary in boundaries:
        start = max(0, int(boundary - beat_span // 2))
        end = min(len(samples), int(boundary + beat_span // 2))
        features.append(spectral_feature(samples[start:end], sample_rate))
    return np.vstack(features)


def build_transition_graph(features, measure_beats, max_candidates):
    total_beats = features.shape[0] - 1
    candidates = []
    all_scores = []
    min_jump = measure_beats * 2

    for current in range(1, total_beats):
        possible = []
        for target in range(1, total_beats):
            if target == current + 1 or abs(target - current) < min_jump:
                continue
            if (target - current) % measure_beats != 0:
                continue
            score = float(np.dot(features[current], features[target]))
            score = max(0.0, min(1.0, score))
            possible.append((score, target))
            all_scores.append(score)
        candidates.append((current, possible))

    if not all_scores:
        return {}, {}, []

    threshold = float(np.percentile(np.asarray(all_scores), 85))
    transitions = {beat: [beat + 1] for beat in range(total_beats)}
    similarities = {}
    transition_points = []

    for current, possible in candidates:
        possible.sort(reverse=True)
        chosen = [item for item in possible if item[0] >= threshold][:max_candidates]
        if not chosen and possible:
            chosen = possible[:1]
        for score, target in chosen:
            if target not in transitions[current]:
                transitions[current].append(target)
                similarities[(current, target)] = score
                transition_points.append((current, target, score))

    return transitions, similarities, transition_points


def path_duration(path):
    return sum(1 for current, target in zip(path[:-1], path[1:]) if target == current + 1)


def path_jumps(path):
    return [(current, target) for current, target in zip(path[:-1], path[1:]) if target != current + 1]


def path_intervals(path):
    intervals = []
    start = path[0]
    previous = path[0]
    for beat in path[1:]:
        if beat == previous + 1:
            previous = beat
            continue
        intervals.append((start, previous))
        start = beat
        previous = beat
    intervals.append((start, previous))
    return [(a, b) for a, b in intervals if b > a]


def construct_preview(samples, boundaries, path, crossfade_seconds, sample_rate):
    intervals = path_intervals(path)
    if not intervals:
        return np.asarray([], dtype=np.float32)

    crossfade = max(0, int(crossfade_seconds * sample_rate))
    output = None
    for start_beat, end_beat in intervals:
        segment = samples[boundaries[start_beat]:boundaries[end_beat]]
        if len(segment) == 0:
            continue
        if output is None:
            output = segment.copy()
            continue
        fade = min(crossfade, len(output), len(segment))
        if fade > 0:
            fade_in = np.linspace(0, 1, fade, endpoint=False, dtype=np.float32)
            fade_out = 1 - fade_in
            output[-fade:] = output[-fade:] * fade_out + segment[:fade] * fade_in
            output = np.concatenate([output, segment[fade:]])
        else:
            output = np.concatenate([output, segment])

    if output is None:
        return np.asarray([], dtype=np.float32)
    peak = float(np.max(np.abs(output))) if len(output) else 0
    if peak > 1:
        output = output / peak
    return output.astype(np.float32)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_bar_chart(path, title, rows, value_key, label_key="track", width=1100, height=420):
    margin_left = 170
    margin_right = 30
    margin_top = 45
    margin_bottom = 45
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max([float(row[value_key]) for row in rows] + [1e-9])
    bar_height = max(8, plot_height / max(1, len(rows)) * 0.65)
    gap = plot_height / max(1, len(rows))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px} .title{font-size:20px;font-weight:700}</style>",
        f'<rect width="{width}" height="{height}" fill="#fbfbf8"/>',
        f'<text class="title" x="{margin_left}" y="28">{html.escape(title)}</text>',
    ]
    for idx, row in enumerate(rows):
        y = margin_top + idx * gap + (gap - bar_height) / 2
        value = float(row[value_key])
        bar_width = plot_width * value / max_value
        label = html.escape(str(row[label_key]))
        parts.append(f'<text x="12" y="{y + bar_height * 0.72}">{label}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2f6f73"/>')
        parts.append(f'<text x="{margin_left + bar_width + 8}" y="{y + bar_height * 0.72}">{value:.3g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_grouped_runtime(path, results, width=1180, height=520):
    tracks = list(dict.fromkeys(row["track"] for row in results))
    ratios = list(dict.fromkeys(row["target_ratio"] for row in results))
    max_runtime = max([row["runtime_ms"] for row in results] + [1])
    margin_left = 150
    margin_top = 55
    margin_bottom = 70
    plot_width = width - margin_left - 35
    plot_height = height - margin_top - margin_bottom
    group_width = plot_width / len(tracks)
    bar_width = min(18, group_width / (len(ratios) + 1))
    colors = ["#2f6f73", "#d08b38", "#6457a6"]

    lookup = {(row["track"], row["target_ratio"]): row for row in results}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}</style>",
        f'<rect width="{width}" height="{height}" fill="#fbfbf8"/>',
        f'<text class="title" x="{margin_left}" y="30">Sparse search runtime by track and target</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 35}" y2="{height - margin_bottom}" stroke="#333"/>',
    ]
    for ratio_idx, ratio in enumerate(ratios):
        x = margin_left + ratio_idx * 155
        parts.append(f'<rect x="{x}" y="40" width="13" height="13" fill="{colors[ratio_idx % len(colors)]}"/>')
        parts.append(f'<text x="{x + 20}" y="51">target {ratio}</text>')
    for track_idx, track in enumerate(tracks):
        base_x = margin_left + track_idx * group_width + group_width * 0.18
        for ratio_idx, ratio in enumerate(ratios):
            row = lookup.get((track, ratio))
            if row is None:
                continue
            h = plot_height * row["runtime_ms"] / max_runtime
            x = base_x + ratio_idx * (bar_width + 3)
            y = height - margin_bottom - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" fill="{colors[ratio_idx % len(colors)]}"/>')
        label = html.escape(track[:16])
        lx = margin_left + track_idx * group_width + group_width * 0.1
        parts.append(f'<text transform="translate({lx:.2f},{height - 48}) rotate(35)">{label}</text>')
    parts.append(f'<text x="{margin_left}" y="{height - 12}">Runtime is just path search, not download/feature extraction.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def svg_timeline(path, track, result, total_beats, width=1100, height=260):
    margin = 55
    plot_width = width - 2 * margin
    mid_y = 130
    scale = plot_width / max(1, total_beats)
    intervals = result["intervals"]
    jumps = result["jumps"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:18px;font-weight:700}</style>",
        f'<rect width="{width}" height="{height}" fill="#fbfbf8"/>',
        f'<text class="title" x="{margin}" y="28">{html.escape(track)}: target {result["target_ratio"]}</text>',
        f'<text x="{margin}" y="49">output {result["output_beats"]} beats, error {result["duration_error_beats"]} beats, {result["n_jumps"]} jumps, {result["runtime_ms"]:.2f} ms</text>',
        f'<line x1="{margin}" y1="{mid_y}" x2="{width - margin}" y2="{mid_y}" stroke="#999" stroke-width="2"/>',
    ]
    for start, end in intervals:
        x1 = margin + start * scale
        x2 = margin + end * scale
        parts.append(f'<line x1="{x1:.2f}" y1="{mid_y}" x2="{x2:.2f}" y2="{mid_y}" stroke="#2f6f73" stroke-width="8" stroke-linecap="round" opacity="0.75"/>')
    for idx, (start, end) in enumerate(jumps[:40]):
        x1 = margin + start * scale
        x2 = margin + end * scale
        up = end > start
        control_y = mid_y - 35 - (idx % 4) * 16 if up else mid_y + 35 + (idx % 4) * 16
        color = "#d08b38" if up else "#6457a6"
        parts.append(f'<path d="M{x1:.2f},{mid_y} Q{(x1 + x2) / 2:.2f},{control_y:.2f} {x2:.2f},{mid_y}" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.78"/>')
    parts.append(f'<text x="{margin}" y="{height - 25}">Green = source intervals used. Orange arcs = forward skips. Purple arcs = backward repeats.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_dashboard(path, track_rows, result_rows):
    rows = []
    for row in result_rows:
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{row['target_ratio']}</td>"
            f"<td>{row['target_beats']}</td>"
            f"<td>{row['output_beats']}</td>"
            f"<td>{row['duration_error_beats']}</td>"
            f"<td>{row['n_jumps']}</td>"
            f"<td>{row['runtime_ms']:.3f}</td>"
            f"<td>{html.escape(row.get('preview', ''))}</td>"
            "</tr>"
        )
    track_json = html.escape(json.dumps(track_rows, indent=2))
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Real audio path-search benchmark</title>"
        "<style>body{font-family:Arial,sans-serif;margin:28px;max-width:1200px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:left}"
        "th{background:#f0f0ec} img{max-width:100%;height:auto}"
        "code,pre{background:#f6f6f3;padding:2px 4px}</style></head><body>"
        "<h1>Real audio path-search benchmark</h1>"
        "<p>Audio is downloaded from Wikimedia Commons. Feature extraction is "
        "a lightweight ffmpeg/numpy harness for exercising the sparse path "
        "search on real MP3 waveforms.</p>"
        "<h2>Plots</h2>"
        "<p><img src='plots/runtime_by_case.svg'></p>"
        "<p><img src='plots/duration_error_by_case.svg'></p>"
        "<p><img src='plots/transition_density.svg'></p>"
        "<h2>Results</h2>"
        "<table><tr><th>Track</th><th>Target</th><th>Target beats</th>"
        "<th>Output beats</th><th>Error</th><th>Jumps</th><th>Runtime ms</th>"
        "<th>Preview</th></tr>"
        + "\n".join(rows)
        + "</table><h2>Track metadata</h2><pre>"
        + track_json
        + "</pre></body></html>"
    )


def run_benchmark(args):
    out_dir = Path(args.output_dir)
    audio_dir = out_dir / "audio"
    preview_dir = out_dir / "previews"
    plot_dir = out_dir / "plots"
    timeline_dir = plot_dir / "timelines"
    for directory in [audio_dir, preview_dir, plot_dir, timeline_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    track_rows = []
    result_rows = []

    for title, filename in TRACKS[:args.limit]:
        slug = safe_name(title)
        audio_path = audio_dir / f"{slug}.mp3"
        print(f"> Downloading {title}")
        download_track(filename, audio_path)

        print(f"  > Decoding and extracting transitions")
        samples = decode_audio(audio_path, args.sample_rate, args.max_seconds)
        if len(samples) < args.sample_rate * 12:
            print(f"  ! Skipping {title}: decoded audio is too short")
            continue

        beat_period, bpm = estimate_beat_period(samples, args.sample_rate)
        boundaries = beat_boundaries(samples, args.sample_rate, beat_period)
        total_beats = len(boundaries) - 1
        measure_beats = args.measure_beats
        features = boundary_features(samples, args.sample_rate, boundaries)
        transitions, similarities, transition_points = build_transition_graph(
            features,
            measure_beats=measure_beats,
            max_candidates=args.max_candidates,
        )

        track_rows.append({
            "track": title,
            "filename": filename,
            "source": commons_page_url(filename),
            "local_audio": str(audio_path),
            "seconds_analyzed": round(len(samples) / args.sample_rate, 3),
            "estimated_bpm": round(bpm, 2),
            "total_beats": total_beats,
            "transition_edges": len(transition_points),
            "transition_edges_per_beat": round(len(transition_points) / max(1, total_beats), 3),
        })

        preview_rendered = False
        for ratio in args.target_ratios:
            target_beats = max(measure_beats * 2, int(round(total_beats * ratio)))
            start_time = time.perf_counter()
            path = sparse_dijkstra_search(
                transitions=transitions,
                similarities=similarities,
                target_beats=target_beats,
                total_beats=total_beats,
                beat_analysis=[[0, beat + 1] for beat in range(measure_beats)],
                tolerance_beats=args.tolerance_beats,
                max_tolerance_beats=args.max_tolerance_beats,
            )
            runtime_ms = (time.perf_counter() - start_time) * 1000
            output_beats = path_duration(path)
            jumps = path_jumps(path)
            intervals = path_intervals(path)
            result = {
                "track": title,
                "target_ratio": ratio,
                "target_beats": target_beats,
                "output_beats": output_beats,
                "duration_error_beats": abs(output_beats - target_beats),
                "n_jumps": len(jumps),
                "runtime_ms": runtime_ms,
                "path_length": len(path),
                "intervals": intervals,
                "jumps": jumps,
                "preview": "",
            }

            if args.render_previews and not preview_rendered and abs(ratio - args.preview_ratio) < 1e-9:
                preview = construct_preview(
                    samples=samples,
                    boundaries=boundaries,
                    path=path,
                    crossfade_seconds=args.crossfade,
                    sample_rate=args.sample_rate,
                )
                preview_path = preview_dir / f"{slug}_target_{str(ratio).replace('.', '_')}.mp3"
                encode_mp3(preview, args.sample_rate, preview_path)
                result["preview"] = str(preview_path.relative_to(out_dir))
                preview_rendered = True

            timeline_path = timeline_dir / f"{slug}_target_{str(ratio).replace('.', '_')}.svg"
            svg_timeline(timeline_path, title, result, total_beats)
            result["timeline"] = str(timeline_path.relative_to(out_dir))

            print(
                f"  > target {ratio}: {output_beats}/{target_beats} beats, "
                f"{len(jumps)} jumps, {runtime_ms:.3f} ms"
            )
            result_rows.append(result)

    csv_rows = []
    for row in result_rows:
        flat = dict(row)
        flat["intervals"] = json.dumps(flat["intervals"])
        flat["jumps"] = json.dumps(flat["jumps"])
        csv_rows.append(flat)

    write_csv(
        out_dir / "results.csv",
        csv_rows,
        [
            "track",
            "target_ratio",
            "target_beats",
            "output_beats",
            "duration_error_beats",
            "n_jumps",
            "runtime_ms",
            "path_length",
            "preview",
            "timeline",
            "intervals",
            "jumps",
        ],
    )
    write_csv(
        out_dir / "tracks.csv",
        track_rows,
        [
            "track",
            "filename",
            "source",
            "local_audio",
            "seconds_analyzed",
            "estimated_bpm",
            "total_beats",
            "transition_edges",
            "transition_edges_per_beat",
        ],
    )

    svg_grouped_runtime(plot_dir / "runtime_by_case.svg", result_rows)
    error_rows = [
        {
            "track": f"{row['track']} {row['target_ratio']}",
            "duration_error_beats": row["duration_error_beats"],
        }
        for row in result_rows
    ]
    svg_bar_chart(
        plot_dir / "duration_error_by_case.svg",
        "Absolute duration error after fallback",
        error_rows,
        "duration_error_beats",
    )
    svg_bar_chart(
        plot_dir / "transition_density.svg",
        "Audio-derived transition edges per beat",
        track_rows,
        "transition_edges_per_beat",
    )
    write_dashboard(out_dir / "index.html", track_rows, result_rows)
    print(f"> Wrote benchmark dashboard to {out_dir / 'index.html'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/real_audio_benchmark")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--sample-rate", type=int, default=11025)
    parser.add_argument("--measure-beats", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--target-ratios", type=float, nargs="+", default=[0.65, 0.8, 1.2])
    parser.add_argument("--tolerance-beats", type=int, default=0)
    parser.add_argument("--max-tolerance-beats", type=int, default=32)
    parser.add_argument("--crossfade", type=float, default=0.06)
    parser.add_argument("--preview-ratio", type=float, default=0.8)
    parser.add_argument("--no-render-previews", action="store_false", dest="render_previews")
    parser.set_defaults(render_previews=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
