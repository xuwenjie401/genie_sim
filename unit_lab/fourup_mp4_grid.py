#!/usr/bin/env python3
"""Compose four MP4 files into a 1920x1080 2x2 grid.

Each input is scaled to fit within a 960x540 cell while preserving aspect
ratio. Inputs that end before the longest clip are extended by freezing their
final frame. The result is written as an H.264 MP4 file without audio.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
GRID_COLUMNS = 2
GRID_ROWS = 2
CELL_WIDTH = FRAME_WIDTH // GRID_COLUMNS
CELL_HEIGHT = FRAME_HEIGHT // GRID_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Arrange four MP4 files evenly in a 1080p 2x2 grid and freeze any "
            "shorter input on its last frame until the longest clip ends."
        ),
        epilog=(
            "Example:\n"
            "  conda run -n issac python unit_lab/fourup_mp4_grid.py "
            "a.mp4 b.mp4 c.mp4 d.mp4 output.mp4 --overwrite"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs=4,
        metavar="INPUT",
        help="Exactly four input .mp4 files.",
    )
    parser.add_argument("output", help="Output .mp4 file.")
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output frame rate. Defaults to 30.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF value. Lower is higher quality. Defaults to 18.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset. Defaults to medium.",
    )
    parser.add_argument(
        "--background",
        default="black",
        help="Pad/background color. Defaults to black.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to ffmpeg. Defaults to ffmpeg from PATH.",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="Path to ffprobe. Defaults to ffprobe from PATH.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than 0.")
    if args.crf < 0:
        parser.error("--crf must be non-negative.")

    return args


def ensure_inputs_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))


def ensure_output_path(output_path: Path) -> None:
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output must end with .mp4: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def probe_duration(input_path: Path, ffprobe_bin: str) -> float:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {input_path}:\n{result.stderr.strip() or result.stdout.strip()}"
        )

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse duration for {input_path}: {result.stdout.strip()!r}"
        ) from exc

    if duration <= 0:
        raise RuntimeError(f"Invalid duration for {input_path}: {duration}")
    return duration


def compute_freeze_durations(durations: list[float]) -> tuple[list[float], float]:
    if len(durations) != 4:
        raise ValueError(f"Expected 4 durations, got {len(durations)}")

    target_duration = max(durations)
    return [max(0.0, target_duration - duration) for duration in durations], target_duration


def build_filter_complex(
    freeze_durations: list[float],
    fps: float,
    background: str,
    target_duration: float,
) -> str:
    if len(freeze_durations) != 4:
        raise ValueError(f"Expected 4 freeze durations, got {len(freeze_durations)}")

    frame_duration = 1.0 / fps
    filters: list[str] = []
    for index, freeze_duration in enumerate(freeze_durations):
        chain = [
            f"[{index}:v]setpts=PTS-STARTPTS",
            (
                "scale="
                f"w={CELL_WIDTH}:h={CELL_HEIGHT}:"
                "force_original_aspect_ratio=decrease"
            ),
            f"pad={CELL_WIDTH}:{CELL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color={background}",
        ]
        if freeze_duration > 0:
            chain.append(f"tpad=stop_mode=clone:stop_duration={freeze_duration:.6f}")
        chain.append(f"setsar=1[v{index}]")
        filters.append(",".join(chain))

    filters.append(
        f"[v0][v1][v2][v3]xstack="
        f"inputs=4:layout=0_0|{CELL_WIDTH}_0|0_{CELL_HEIGHT}|{CELL_WIDTH}_{CELL_HEIGHT}:"
        f"fill={background}[stack]"
    )
    filters.append(
        f"[stack]fps=fps={fps:g}:round=up,"
        f"tpad=stop_mode=clone:stop_duration={frame_duration:.6f},"
        f"trim=duration={target_duration:.6f},"
        f"setpts=PTS-STARTPTS,"
        "format=yuv420p[vout]"
    )
    return ";".join(filters)


def build_ffmpeg_command(
    input_paths: list[Path],
    output_path: Path,
    ffmpeg_bin: str,
    filter_complex: str,
    preset: str,
    crf: int,
    overwrite: bool,
) -> list[str]:
    if len(input_paths) != 4:
        raise ValueError(f"Expected 4 inputs, got {len(input_paths)}")

    command = [ffmpeg_bin, "-hide_banner", "-y" if overwrite else "-n"]
    for input_path in input_paths:
        command.extend(["-i", str(input_path)])

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return command


def main() -> int:
    args = parse_args()
    input_paths = [Path(path).expanduser().resolve() for path in args.inputs]
    output_path = Path(args.output).expanduser().resolve()

    ensure_inputs_exist(input_paths)
    ensure_output_path(output_path)

    durations = [probe_duration(path, args.ffprobe) for path in input_paths]
    freeze_durations, target_duration = compute_freeze_durations(durations)
    filter_complex = build_filter_complex(
        freeze_durations=freeze_durations,
        fps=args.fps,
        background=args.background,
        target_duration=target_duration,
    )
    command = build_ffmpeg_command(
        input_paths=input_paths,
        output_path=output_path,
        ffmpeg_bin=args.ffmpeg,
        filter_complex=filter_complex,
        preset=args.preset,
        crf=args.crf,
        overwrite=args.overwrite,
    )

    duration_text = ", ".join(f"{duration:.2f}s" for duration in durations)
    print(f"Input durations: {duration_text}", file=sys.stderr)
    print(f"Output duration: {target_duration:.2f}s", file=sys.stderr)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
