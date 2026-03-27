#!/usr/bin/env python3
"""Visualize and sanity-check one extracted recording_data episode.

Usage:
    python source/data_collection/scripts/visualize_recording_data.py \
        --dir /abs/path/to/recording_data/<episode_dir>

Dependencies:
    - tkinter (usually available with Python)
    - Pillow (`pip install Pillow`) for image rendering in the viewer
    - ffmpeg/ffprobe in PATH when using MP4-backed playback

This viewer uses `state.json` as the frame source for joints and end-effector
poses. For image playback it prefers extracted `camera/<frame_id>/...jpg`
frames, and falls back to `observations/videos/*.mp4` when raw frame folders
are absent. It also prints MP4 frame counts via `ffprobe` when videos exist.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install it with: python -m pip install Pillow"
    ) from exc


IMAGE_CANDIDATES = {
    "head": ["head_color.jpg", "head_front_left_color.jpg", "head_front_color.jpg"],
    "left": ["hand_left_color.jpg", "left_arm_color.jpg"],
    "right": ["hand_right_color.jpg", "right_arm_color.jpg"],
}

VIDEO_CANDIDATES = {
    "head": [
        "head_color.mp4",
        "head_front_color.mp4",
        "head_front_left_color.mp4",
        "head_front_left_color_color.mp4",
    ],
    "left": [
        "hand_left_color.mp4",
        "left_arm_color.mp4",
        "left_arm_color_color.mp4",
    ],
    "right": [
        "hand_right_color.mp4",
        "right_arm_color.mp4",
        "right_arm_color_color.mp4",
    ],
}

# Fonts ordered by preference; X11-native names come first because modern
# fontconfig names (Cabin, DejaVu, …) fall back to ugly "fixed" bitmap on
# systems where Tk was compiled without fontconfig/Xft support.
# "Helvetica" aliases to the X11 "gothic" font: clean sans-serif, uniform
# stroke width (no bold/thin serifs), much more readable than bitmap "fixed".
CUTE_FONT_CANDIDATES = [
    "Helvetica",          # X11 gothic sans-serif — uniform strokes, clean look
    "clearlyu",           # X11 unicode sans fallback
    "bitstream charter",  # serif fallback
    "Cabin",
    "Noto Sans",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
]

# Monospace font candidates; tried in order, first working one is used.
MONO_FONT_CANDIDATES = [
    "courier 10 pitch",  # proper proportional-mono, X11 native
    "Courier New",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier",
]

DEFAULT_UI_FONT_FAMILY = "Helvetica"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one recording_data directory.")
    parser.add_argument(
        "--dir",
        "--recording-dir",
        dest="recording_dir",
        required=False,
        help="Path to one recording_data episode directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to recording_info.json fps, else 30.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=420,
        help="Rendered width for each camera image pane.",
    )
    parser.add_argument(
        "--font-family",
        default=None,
        help="Optional UI font family override, for example 'Comic Neue' or 'Chilanka'.",
    )
    parser.add_argument(
        "--list-fonts",
        action="store_true",
        help="Print Tk-visible font families and exit.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _state_chunk_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("state_"):
        suffix = stem.split("_", 1)[1]
        if suffix.isdigit():
            return (int(suffix), path.name)
    return (sys.maxsize, path.name)


def load_state_frames(recording_dir: Path) -> tuple[list[dict[str, Any]], str]:
    state_path = recording_dir / "state.json"
    if state_path.is_file():
        state = load_json(state_path)
        frames = state.get("frames", [])
        return frames, str(state_path)

    chunk_paths = sorted(recording_dir.glob("state_*.json"), key=_state_chunk_sort_key)
    if not chunk_paths:
        raise SystemExit(f"Missing required file: {state_path}")

    frames: list[dict[str, Any]] = []
    loaded_chunks = 0
    skipped_chunks: list[str] = []
    for chunk_path in chunk_paths:
        try:
            chunk_state = load_json(chunk_path)
        except json.JSONDecodeError as exc:
            skipped_chunks.append(f"{chunk_path.name} ({exc})")
            continue
        frames.extend(chunk_state.get("frames", []))
        loaded_chunks += 1

    if not frames:
        skipped = ", ".join(skipped_chunks) if skipped_chunks else "none"
        raise SystemExit(
            f"Failed to load usable frames from chunked state files in {recording_dir}. "
            f"Skipped: {skipped}"
        )

    if skipped_chunks:
        print("warning: state.json missing; using chunked state files")
        for skipped in skipped_chunks:
            print(f"warning: skipped corrupted chunk: {skipped}")

    source_desc = f"{loaded_chunks} chunk file(s)"
    return frames, source_desc


def decode_names(values: Any) -> list[str]:
    names: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def load_joint_data(recording_dir: Path) -> tuple[list[str], np.ndarray]:
    h5_path = recording_dir / "aligned_joints_all.h5"
    if not h5_path.is_file():
        return [], np.zeros((0, 0), dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        if "state/joint/position" not in f:
            return [], np.zeros((0, 0), dtype=np.float32)
        joint_names = decode_names(f["state/joint"].attrs.get("name", []))
        joint_values = f["state/joint/position"][:]
    return joint_names, np.asarray(joint_values, dtype=np.float32)


def load_task_description(recording_dir: Path, recording_info: dict[str, Any]) -> dict[str, str]:
    task_description = {
        "task_name": str(recording_info.get("task_name", "") or ""),
        "english_task_name": "",
        "init_scene_text": "",
    }

    frame_state_path = recording_dir / "frame_state.json"
    if not frame_state_path.is_file():
        return task_description

    try:
        frame_states = load_json(frame_state_path)
    except json.JSONDecodeError:
        return task_description

    if not isinstance(frame_states, list):
        return task_description

    for frame_state in frame_states:
        candidate = frame_state.get("task_description")
        if not isinstance(candidate, dict):
            continue
        for key in ("task_name", "english_task_name", "init_scene_text"):
            value = candidate.get(key)
            if value:
                task_description[key] = str(value)
        if any(task_description.get(key) for key in ("task_name", "english_task_name", "init_scene_text")):
            break

    return task_description


def format_xyz_quat_wxyz(pose_4x4: list[list[float]]) -> str:
    x = pose_4x4[0][3]
    y = pose_4x4[1][3]
    z = pose_4x4[2][3]

    r11, r12, r13 = pose_4x4[0][:3]
    r21, r22, r23 = pose_4x4[1][:3]
    r31, r32, r33 = pose_4x4[2][:3]
    trace = r11 + r22 + r33
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r32 - r23) * s
        qy = (r13 - r31) * s
        qz = (r21 - r12) * s
    elif r11 > r22 and r11 > r33:
        s = 2.0 * np.sqrt(max(1.0 + r11 - r22 - r33, 0.0))
        qw = (r32 - r23) / s
        qx = 0.25 * s
        qy = (r12 + r21) / s
        qz = (r13 + r31) / s
    elif r22 > r33:
        s = 2.0 * np.sqrt(max(1.0 + r22 - r11 - r33, 0.0))
        qw = (r13 - r31) / s
        qx = (r12 + r21) / s
        qy = 0.25 * s
        qz = (r23 + r32) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + r33 - r11 - r22, 0.0))
        qw = (r21 - r12) / s
        qx = (r13 + r31) / s
        qy = (r23 + r32) / s
        qz = 0.25 * s

    return (
        f"xyz: [{x: .4f}, {y: .4f}, {z: .4f}]\n"
        f"quat(wxyz): [{qw: .4f}, {qx: .4f}, {qy: .4f}, {qz: .4f}]"
    )


def split_left_right_joints(joint_names: list[str], joint_values: list[float]) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    left: list[tuple[str, float]] = []
    right: list[tuple[str, float]] = []

    for name, value in zip(joint_names, joint_values):
        lower = name.lower()

        # Prefer explicit arm/side markers over generic subcomponent markers like
        # "_l_" / "_r_" so names such as "right_gripper_l_knuckle_joint" stay
        # grouped under the right side.
        if (
            lower.startswith("fr_")
            or lower.startswith("right_")
            or "right" in lower
            or "_right_" in lower
        ):
            right.append((name, value))
        elif (
            lower.startswith("fl_")
            or lower.startswith("left_")
            or "left" in lower
            or "_left_" in lower
        ):
            left.append((name, value))
        elif lower.startswith("r_") or "_r_" in lower:
            right.append((name, value))
        elif lower.startswith("l_") or "_l_" in lower:
            left.append((name, value))

    if left or right:
        return left, right

    midpoint = len(joint_names) // 2
    return (
        list(zip(joint_names[:midpoint], joint_values[:midpoint])),
        list(zip(joint_names[midpoint:], joint_values[midpoint:])),
    )


def count_mp4_frames(video_path: Path) -> str:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def collect_video_counts(recording_dir: Path) -> dict[str, str]:
    videos_dir = recording_dir / "observations" / "videos"
    if not videos_dir.is_dir():
        return {}
    counts = {}
    for video_path in sorted(videos_dir.glob("*.mp4")):
        counts[video_path.name] = count_mp4_frames(video_path)
    return counts


def find_image_for_frame(camera_dir: Path, logical_name: str) -> Path | None:
    for candidate in IMAGE_CANDIDATES[logical_name]:
        candidate_path = camera_dir / candidate
        if candidate_path.is_file():
            return candidate_path
    return None


def find_video_for_logical_name(videos_dir: Path, logical_name: str) -> Path | None:
    for candidate in VIDEO_CANDIDATES[logical_name]:
        candidate_path = videos_dir / candidate
        if candidate_path.is_file():
            return candidate_path
    return None


def probe_video_size(video_path: Path) -> tuple[int, int] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout or "{}")
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    return width, height


class FFmpegVideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.size = probe_video_size(video_path)
        self.width = self.size[0] if self.size else 0
        self.height = self.size[1] if self.size else 0
        self.frame_bytes = self.width * self.height * 3
        self.process: subprocess.Popen[bytes] | None = None
        self.next_frame_idx = 0
        self.cache: dict[int, Image.Image] = {}
        self.cache_order: list[int] = []
        self.cache_limit = 8

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdout is not None:
            self.process.stdout.close()
        try:
            self.process.terminate()
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)
        finally:
            self.process = None

    def read_frame(self, frame_idx: int) -> Image.Image | None:
        if frame_idx < 0 or self.frame_bytes <= 0:
            return None
        cached = self.cache.get(frame_idx)
        if cached is not None:
            return cached.copy()
        if self.process is None or frame_idx < self.next_frame_idx:
            if not self._start():
                return None
        while self.next_frame_idx <= frame_idx:
            frame = self._read_one_frame()
            if frame is None:
                return None
            current_idx = self.next_frame_idx
            self.next_frame_idx += 1
            self._cache_frame(current_idx, frame)
            if current_idx == frame_idx:
                return frame
        return None

    def _start(self) -> bool:
        self.close()
        if self.frame_bytes <= 0:
            return False
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(self.video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "0",
            "-",
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.process = None
            return False
        self.next_frame_idx = 0
        return True

    def _read_one_frame(self) -> Image.Image | None:
        if self.process is None or self.process.stdout is None:
            return None
        payload = self._read_exact(self.frame_bytes)
        if payload is None:
            self.close()
            return None
        return Image.frombytes("RGB", (self.width, self.height), payload)

    def _read_exact(self, size: int) -> bytes | None:
        if self.process is None or self.process.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _cache_frame(self, frame_idx: int, image: Image.Image) -> None:
        self.cache[frame_idx] = image.copy()
        self.cache_order.append(frame_idx)
        while len(self.cache_order) > self.cache_limit:
            stale_idx = self.cache_order.pop(0)
            self.cache.pop(stale_idx, None)


class RecordingViewer:
    def __init__(
        self,
        recording_dir: Path,
        frame_data: list[dict[str, Any]],
        playback_fps: float,
        image_width: int,
        mp4_counts: dict[str, str],
        joint_names: list[str],
        joint_values: np.ndarray,
        task_description: dict[str, str],
        font_family_override: str | None,
    ):
        self.recording_dir = recording_dir
        self.frame_data = frame_data
        self.base_playback_fps = max(float(playback_fps), 0.1)
        self.play_rate = 1.0
        self.playback_fps = self.base_playback_fps * self.play_rate
        self.image_width = image_width
        self.mp4_counts = mp4_counts
        self.joint_names = joint_names
        self.joint_values = joint_values
        self.task_description = task_description
        self.frame_count = len(frame_data)
        self.pose_plot_limits = self._compute_pose_plot_limits()
        self.frame_idx = 0
        self.playing = True
        self.after_id: str | None = None
        self._last_advance_time = time.monotonic()
        self._frame_credit = 0.0
        self._image_refs: list[ImageTk.PhotoImage] = []

        self.root = tk.Tk()
        self.root.title(f"Recording Viewer: {recording_dir.name}")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(1800, max(screen_w - 80, 1200))
        height = min(1080, max(screen_h - 120, 800))
        self.root.geometry(f"{width}x{height}")
        self.pose_image_size = max(170, min(240, int(height * 0.19)))

        self.root.configure(bg="#f5f1e8")
        self.ui_font_family = self._pick_ui_font_family(font_family_override)
        self.mono_font_family = self._pick_mono_font_family()
        self.videos_dir = self.recording_dir / "observations" / "videos"
        self.video_paths = {
            logical_name: find_video_for_logical_name(self.videos_dir, logical_name)
            for logical_name in ("head", "left", "right")
        }
        self.video_readers = {
            logical_name: (FFmpegVideoReader(video_path) if video_path is not None else None)
            for logical_name, video_path in self.video_paths.items()
        }
        self.image_labels: dict[str, tk.Label] = {}
        self.pose_image_labels: dict[str, tk.Label] = {}
        self.text_widgets: dict[str, tk.Text] = {}
        self.pose_numeric_labels: dict[str, tk.Label] = {}
        self.robot_pose_labels: dict[str, tk.Label] = {}
        self.status_var = tk.StringVar()
        self.counts_var = tk.StringVar()

        self._configure_fonts()
        self._build_ui()
        self._bind_keys()
        self.render_frame()
        self.schedule_next()

    def _update_playback_fps(self) -> None:
        self.playback_fps = max(self.base_playback_fps * self.play_rate, 0.1)

    def _playback_summary(self) -> str:
        state = "playing" if self.playing else "paused"
        return (
            f"rate: {self.play_rate:.2f}x | "
            f"effective_fps: {self.playback_fps:.2f} | "
            f"state: {state}"
        )

    def _refresh_header_counts(self) -> None:
        counts = (
            f"state/joints/eef frames: {self.frame_count} | "
            f"camera frame dirs: {self._count_camera_dirs()} | "
            f"mp4 panes: {self._count_available_videos()}/3 | "
            f"mono: {self.mono_font_family}"
        )
        if self.mp4_counts:
            video_counts = " | ".join(
                f"{name}: {count}" for name, count in sorted(self.mp4_counts.items())
            )
            counts = f"{counts}\nmp4 frames: {video_counts}"
        counts = (
            f"{counts}\n{self._playback_summary()}"
            "\ncontrols: Space play/pause, Left/Right step, Up/+/= faster, Down/- slower, Esc quit"
        )
        self.counts_var.set(counts)

    def _configure_fonts(self) -> None:
        # Don't touch font families at all — this Tk has no Xft, so any named
        # family (Cabin, Helvetica, …) maps to "fixed" or "gothic" bitmap.
        # Only adjust sizes on the system-default fonts so they're readable.
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=13)

        # Derive named font objects from the system default so family is whatever
        # the desktop actually configured (usually a clean sans-serif).
        base = default_font.actual()
        self._font_title   = tkfont.Font(family=base["family"], size=16, weight="bold")
        self._font_heading = tkfont.Font(family=base["family"], size=13, weight="bold")
        self._font_label   = tkfont.Font(family=base["family"], size=13)
        self._font_small   = tkfont.Font(family=base["family"], size=11)
        self._font_tiny    = tkfont.Font(family=base["family"], size=10)
        self._font_mono    = tkfont.Font(family=self.mono_font_family, size=12)
        self._font_mono_sm = tkfont.Font(family=self.mono_font_family, size=11)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f5f1e8")

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#f5f1e8", padx=10, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            header,
            text=self.recording_dir.name,
            font=self._font_title,
            bg="#f5f1e8",
            fg="#1f1f1f",
        ).pack(anchor="w")
        self._refresh_header_counts()
        tk.Label(
            header,
            textvariable=self.counts_var,
            justify=tk.LEFT,
            font=self._font_label,
            bg="#f5f1e8",
            fg="#2f2f2f",
        ).pack(anchor="w", pady=(6, 0))
        tk.Label(
            header,
            text=(
                "EEF pose declaration:\n"
                "left_eef_pose = T_world_left-tcp-link\n"
                "right_eef_pose = T_world_right-tcp-link"
            ),
            justify=tk.LEFT,
            font=self._font_small,
            bg="#f5f1e8",
            fg="#4a4036",
        ).pack(anchor="w", pady=(8, 0))
        task_instruction = self._format_task_instruction()
        if task_instruction:
            tk.Label(
                header,
                text=task_instruction,
                justify=tk.LEFT,
                font=self._font_small,
                bg="#f5f1e8",
                fg="#3b342d",
            ).pack(anchor="w", pady=(8, 0))

        robot_info_frame = tk.Frame(self.root, bg="#f5f1e8", padx=8, pady=2)
        robot_info_frame.pack(side=tk.TOP, fill=tk.X)
        # NOTE: codex arm_base
        robot_pose_panels = (
            ("world_base_link", "T_world_base_link"),
            ("world_arm_base", "T_world_arm_base(shared)"),
            ("world_left_arm_base", "T_world_left_arm_base"),
            ("world_right_arm_base", "T_world_right_arm_base"),
        )
        for idx, (key, title) in enumerate(robot_pose_panels):
            panel = tk.Frame(robot_info_frame, bg="#f5f1e8", padx=6, pady=4)
            panel.grid(row=0, column=idx, sticky="nsew")
            tk.Label(
                panel,
                text=title,
                font=self._font_heading,
                bg="#f5f1e8",
                fg="#1f1f1f",
            ).pack(anchor="w")
            label = tk.Label(
                panel,
                text="",
                justify=tk.LEFT,
                font=self._font_mono_sm,
                bg="#fffdf7",
                fg="#1f1f1f",
                relief=tk.SOLID,
                bd=1,
                padx=8,
                pady=4,
                anchor="w",
            )
            label.pack(anchor="w", fill=tk.X)
            self.robot_pose_labels[key] = label
        robot_info_frame.grid_columnconfigure(0, weight=1)
        robot_info_frame.grid_columnconfigure(1, weight=1)
        robot_info_frame.grid_columnconfigure(2, weight=1)
        robot_info_frame.grid_columnconfigure(3, weight=1)

        image_frame = tk.Frame(self.root, bg="#f5f1e8", padx=8, pady=8)
        image_frame.pack(side=tk.TOP, fill=tk.X)

        for idx, logical_name in enumerate(("head", "left", "right")):
            panel = tk.Frame(image_frame, bg="#f5f1e8", padx=6, pady=6)
            panel.grid(row=0, column=idx, sticky="nsew")
            tk.Label(
                panel,
                text=f"{logical_name}_image",
                font=self._font_heading,
                bg="#f5f1e8",
                fg="#1f1f1f",
            ).pack(anchor="w")
            label = tk.Label(panel, bg="#ddd8cd", bd=1, relief=tk.SOLID)
            label.pack()
            self.image_labels[logical_name] = label
        for idx in range(3):
            image_frame.grid_columnconfigure(idx, weight=1)

        middle_frame = tk.Frame(self.root, bg="#f5f1e8", padx=8, pady=8)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        pose_frame = tk.Frame(middle_frame, bg="#f5f1e8", padx=4, pady=4)
        pose_frame.grid(row=0, column=0, sticky="nsew")
        joints_frame = tk.Frame(middle_frame, bg="#f5f1e8", padx=4, pady=4)
        joints_frame.grid(row=0, column=1, sticky="nsew")
        middle_frame.grid_columnconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(1, weight=1)
        middle_frame.grid_rowconfigure(0, weight=1)

        for idx, key in enumerate(("left_eef_pose", "right_eef_pose")):
            panel = tk.Frame(pose_frame, bg="#f5f1e8", padx=6, pady=6)
            panel.grid(row=0, column=idx, sticky="nsew")
            tk.Label(
                panel,
                text=key,
                font=self._font_heading,
                bg="#f5f1e8",
                fg="#1f1f1f",
            ).pack(anchor="w")
            pose_image = tk.Label(panel, bg="#ffffff", bd=1, relief=tk.SOLID)
            pose_image.pack()
            numeric_label = tk.Label(
                panel,
                text="",
                justify=tk.LEFT,
                font=self._font_mono_sm,
                bg="#f5f1e8",
                fg="#1f1f1f",
                anchor="w",
            )
            numeric_label.pack(anchor="w", pady=(8, 0))
            tk.Label(
                panel,
                text="RGB axes move inside a fixed world-frame range.",
                justify=tk.LEFT,
                font=self._font_tiny,
                bg="#f5f1e8",
                fg="#5a4f43",
            ).pack(anchor="w", pady=(4, 0))
            self.pose_image_labels[key] = pose_image
            self.pose_numeric_labels[key] = numeric_label
        pose_frame.grid_columnconfigure(0, weight=1)
        pose_frame.grid_columnconfigure(1, weight=1)

        for idx, key in enumerate(("left_joints", "right_joints")):
            panel = tk.Frame(joints_frame, bg="#f5f1e8", padx=6, pady=6)
            panel.grid(row=0, column=idx, sticky="nsew")
            tk.Label(
                panel,
                text=key,
                font=self._font_heading,
                bg="#f5f1e8",
                fg="#1f1f1f",
            ).pack(anchor="w")
            text = tk.Text(
                panel,
                width=40,
                height=16,
                wrap="none",
                font=self._font_mono,
                bg="#fffdf7",
                fg="#1f1f1f",
                insertbackground="#1f1f1f",
                relief=tk.SOLID,
                bd=1,
            )
            text.pack(fill=tk.BOTH, expand=True)
            self.text_widgets[key] = text
        joints_frame.grid_columnconfigure(0, weight=1)
        joints_frame.grid_columnconfigure(1, weight=1)

        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=self._font_small,
            bg="#e6dfd1",
            fg="#1f1f1f",
            padx=8,
            pady=8,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", self._toggle_play)
        self.root.bind("<Right>", self._next_frame)
        self.root.bind("<Left>", self._prev_frame)
        self.root.bind("<Up>", self._increase_rate)
        self.root.bind("<Down>", self._decrease_rate)
        self.root.bind("<plus>", self._increase_rate)
        self.root.bind("<equal>", self._increase_rate)
        self.root.bind("<minus>", self._decrease_rate)
        self.root.bind("<Escape>", self._quit)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _toggle_play(self, _event: tk.Event | None = None) -> None:
        self.playing = not self.playing
        self._last_advance_time = time.monotonic()
        self._frame_credit = 0.0
        self._refresh_header_counts()
        self.render_frame()
        self.schedule_next()

    def _next_frame(self, _event: tk.Event | None = None) -> None:
        self.frame_idx = min(self.frame_idx + 1, self.frame_count - 1)
        self.render_frame()

    def _prev_frame(self, _event: tk.Event | None = None) -> None:
        self.frame_idx = max(self.frame_idx - 1, 0)
        self.render_frame()

    def _increase_rate(self, _event: tk.Event | None = None) -> None:
        self.play_rate = min(self.play_rate * 1.25, 16.0)
        self._update_playback_fps()
        self._last_advance_time = time.monotonic()
        self._refresh_header_counts()
        self.render_frame()
        self.schedule_next()

    def _decrease_rate(self, _event: tk.Event | None = None) -> None:
        self.play_rate = max(self.play_rate / 1.25, 0.1)
        self._update_playback_fps()
        self._last_advance_time = time.monotonic()
        self._refresh_header_counts()
        self.render_frame()
        self.schedule_next()

    def _quit(self, _event: tk.Event | None = None) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        for reader in self.video_readers.values():
            if reader is not None:
                reader.close()
        self.root.destroy()

    def schedule_next(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.playing and self.frame_count > 0:
            # Drive playback from elapsed wall time instead of one frame per tick,
            # so rates above 1.0x can still skip frames when rendering is slower
            # than the requested playback speed.
            delay_ms = 16
            self.after_id = self.root.after(delay_ms, self._advance)

    def _advance(self) -> None:
        if self.playing:
            now = time.monotonic()
            elapsed = max(now - self._last_advance_time, 0.0)
            self._last_advance_time = now
            self._frame_credit += elapsed * self.playback_fps
            frame_step = max(int(self._frame_credit), 1)
            self._frame_credit = max(self._frame_credit - frame_step, 0.0)
            self.frame_idx = (self.frame_idx + frame_step) % self.frame_count
            self.render_frame()
        self.schedule_next()

    def render_frame(self) -> None:
        frame = self.frame_data[self.frame_idx]
        self._image_refs = []

        for logical_name in ("head", "left", "right"):
            image = self._load_frame_image(logical_name, self.frame_idx)
            self._render_image(logical_name, image)

        joint_names = self.joint_names
        joint_values = self._get_joint_values_for_frame(self.frame_idx)
        left_joints, right_joints = split_left_right_joints(joint_names, joint_values)

        left_pose_text = format_xyz_quat_wxyz(frame["ee"]["left"]["pose"])
        right_pose_text = format_xyz_quat_wxyz(frame["ee"]["right"]["pose"])
        self._render_pose_image("left_eef_pose", frame["ee"]["left"]["pose"])
        self._render_pose_image("right_eef_pose", frame["ee"]["right"]["pose"])
        self.pose_numeric_labels["left_eef_pose"].configure(text=left_pose_text)
        self.pose_numeric_labels["right_eef_pose"].configure(text=right_pose_text)
        # NOTE: codex arm_base
        self.robot_pose_labels["world_base_link"].configure(
            text=self._format_robot_pose_label(frame, "pose")
        )
        self.robot_pose_labels["world_arm_base"].configure(
            text=self._format_robot_pose_label(frame, "arm_base_pose", "pose")
        )
        self.robot_pose_labels["world_left_arm_base"].configure(
            text=self._format_robot_pose_label(frame, "left_arm_base_pose", "arm_base_pose")
        )
        self.robot_pose_labels["world_right_arm_base"].configure(
            text=self._format_robot_pose_label(frame, "right_arm_base_pose", "arm_base_pose")
        )
        self._set_text("left_joints", self._format_joint_block(left_joints))
        self._set_text("right_joints", self._format_joint_block(right_joints))

        self.status_var.set(
            f"frame {self.frame_idx + 1}/{self.frame_count} | "
            f"time_stamp={frame.get('time_stamp', 'n/a'):.4f} | "
            f"{self._playback_summary()} | "
            "keys: Space play/pause, Left/Right step, Up/+/= faster, Down/- slower, Esc quit"
        )

    def _load_frame_image(self, logical_name: str, frame_idx: int) -> Image.Image | None:
        camera_dir = self.recording_dir / "camera" / str(frame_idx)
        image_path = find_image_for_frame(camera_dir, logical_name)
        if image_path is not None and image_path.is_file():
            return Image.open(image_path).convert("RGB")
        reader = self.video_readers.get(logical_name)
        if reader is None:
            return None
        return reader.read_frame(frame_idx)

    def _render_image(self, logical_name: str, image: Image.Image | None) -> None:
        label = self.image_labels[logical_name]
        if image is None:
            if hasattr(label, "_base_pil_image"):
                delattr(label, "_base_pil_image")
            label.configure(text="missing image", image="")
            return

        scale = self.image_width / max(image.width, 1)
        image = image.resize((self.image_width, max(int(image.height * scale), 1)))
        label._base_pil_image = image
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo, text="")
        self._image_refs.append(photo)

    def _render_pose_image(self, key: str, pose_4x4: list[list[float]]) -> None:
        label = self.pose_image_labels[key]
        image = self._create_pose_overlay(pose_4x4)
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo, text="")
        self._image_refs.append(photo)

    def _create_pose_overlay(self, pose_4x4: list[list[float]]) -> Image.Image:
        fig_size_inches = self.pose_image_size / 100.0
        fig = plt.figure(figsize=(fig_size_inches, fig_size_inches), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")

        origin = np.array([pose_4x4[0][3], pose_4x4[1][3], pose_4x4[2][3]], dtype=np.float32)
        rotation = np.array([row[:3] for row in pose_4x4[:3]], dtype=np.float32)
        axis_scale = 0.10
        colors = [("#ff4d4f", 0), ("#52c41a", 1), ("#1677ff", 2)]

        ax.scatter([origin[0]], [origin[1]], [origin[2]], c="black", s=18)
        for color, idx in colors:
            axis_end = origin + rotation[:, idx] * axis_scale
            delta = axis_end - origin
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                delta[0],
                delta[1],
                delta[2],
                color=color,
                linewidth=2.5,
                arrow_length_ratio=0.2,
            )

        xlim, ylim, zlim = self.pose_plot_limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=22, azim=-58)
        ax.set_xlabel("X", color="#ff4d4f")
        ax.set_ylabel("Y", color="#52c41a")
        ax.set_zlabel("Z", color="#1677ff")
        ax.grid(False)
        ax.set_box_aspect(
            (
                max(xlim[1] - xlim[0], 1e-6),
                max(ylim[1] - ylim[0], 1e-6),
                max(zlim[1] - zlim[0], 1e-6),
            )
        )
        plt.tight_layout(pad=0.4)

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", transparent=False, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _get_joint_values_for_frame(self, frame_idx: int) -> list[float]:
        if self.joint_values.size == 0:
            return []
        safe_idx = min(frame_idx, self.joint_values.shape[0] - 1)
        return self.joint_values[safe_idx].tolist()

    def _format_joint_block(self, joint_pairs: list[tuple[str, float]]) -> str:
        if not joint_pairs:
            return "No joint values found."
        return "\n".join(f"{name:<18} {value: .5f}" for name, value in joint_pairs)

    def _format_task_instruction(self) -> str:
        task_name = self.task_description.get("task_name", "").strip()
        english_task_name = self.task_description.get("english_task_name", "").strip()
        init_scene_text = self.task_description.get("init_scene_text", "").strip()

        lines: list[str] = []
        if task_name:
            lines.append(f"task(zh): {task_name}")
        if english_task_name:
            lines.append(f"task(en): {english_task_name}")
        if init_scene_text:
            lines.append(f"scene(zh): {init_scene_text}")
        return "\n".join(lines)

    def _format_robot_pose_label(
        self,
        frame: dict[str, Any],
        primary_key: str,
        fallback_key: str | None = None,
    ) -> str:
        # NOTE: codex arm_base
        robot_frame = frame.get("robot", {})
        pose = robot_frame.get(primary_key)
        if pose is None and fallback_key is not None:
            pose = robot_frame.get(fallback_key)
        if pose is None:
            return "missing pose"
        return format_xyz_quat_wxyz(pose)

    def _compute_pose_plot_limits(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        points: list[np.ndarray] = []
        for frame in self.frame_data:
            for side in ("left", "right"):
                pose = frame.get("ee", {}).get(side, {}).get("pose")
                if pose:
                    points.append(np.array([pose[0][3], pose[1][3], pose[2][3]], dtype=np.float32))

        if not points:
            return ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))

        stacked = np.vstack(points)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        center = (mins + maxs) / 2.0
        half_range = float(np.max(maxs - mins) / 2.0)
        half_range = max(half_range * 1.15, 0.12)

        return tuple(
            (float(center[idx] - half_range), float(center[idx] + half_range))
            for idx in range(3)
        )

    def _set_text(self, key: str, content: str) -> None:
        widget = self.text_widgets[key]
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)

    def _count_camera_dirs(self) -> int:
        camera_dir = self.recording_dir / "camera"
        if not camera_dir.is_dir():
            return 0
        return sum(1 for p in camera_dir.iterdir() if p.is_dir() and p.name.isdigit())

    def _count_available_videos(self) -> int:
        return sum(1 for path in self.video_paths.values() if path is not None)

    def _pick_mono_font_family(self) -> str:
        available = set(tkfont.families(self.root))
        normalized = {family.lower(): family for family in available}
        for family in MONO_FONT_CANDIDATES:
            if family in available:
                return family
            if family.lower() in normalized:
                return normalized[family.lower()]
        return "TkFixedFont"

    def _pick_ui_font_family(self, font_family_override: str | None) -> str:
        available = set(tkfont.families(self.root))
        if font_family_override:
            if font_family_override in available:
                return font_family_override
            normalized = {family.lower(): family for family in available}
            if font_family_override.lower() in normalized:
                return normalized[font_family_override.lower()]
        normalized = {family.lower(): family for family in available}
        if DEFAULT_UI_FONT_FAMILY in available:
            return DEFAULT_UI_FONT_FAMILY
        if DEFAULT_UI_FONT_FAMILY.lower() in normalized:
            return normalized[DEFAULT_UI_FONT_FAMILY.lower()]
        for family in CUTE_FONT_CANDIDATES:
            if family in available:
                return family
        for family in CUTE_FONT_CANDIDATES:
            if family.lower() in normalized:
                return normalized[family.lower()]
        return "TkDefaultFont"

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    if args.list_fonts:
        root = tk.Tk()
        root.withdraw()
        for family in sorted(tkfont.families(root)):
            print(family)
        root.destroy()
        return 0
    if not args.recording_dir:
        raise SystemExit("Missing required argument: --dir/--recording-dir")
    recording_dir = Path(args.recording_dir).expanduser().resolve()

    frames, state_source = load_state_frames(recording_dir)
    if not frames:
        raise SystemExit(f"No frames found in {state_source}")

    recording_info_path = recording_dir / "recording_info.json"
    recording_info = load_json(recording_info_path) if recording_info_path.is_file() else {}
    playback_fps = float(args.fps or recording_info.get("fps", 30))
    task_description = load_task_description(recording_dir, recording_info)

    mp4_counts = collect_video_counts(recording_dir)
    joint_names, joint_values = load_joint_data(recording_dir)
    print(f"recording_dir: {recording_dir}")
    print(f"state source: {state_source}")
    print(f"joints/eef/state frame count: {len(frames)}")
    print(f"joint h5 frame count: {joint_values.shape[0] if joint_values.size else 0}")
    if mp4_counts:
        print("mp4 frame counts:")
        for name, count in mp4_counts.items():
            print(f"  {name}: {count}")
    else:
        print("mp4 frame counts: no mp4 files found under observations/videos")

    camera_dir = recording_dir / "camera"
    if camera_dir.is_dir():
        extracted_camera_frames = sum(1 for p in camera_dir.iterdir() if p.is_dir() and p.name.isdigit())
        print(f"camera folder frame count: {extracted_camera_frames}")
    else:
        print("camera folder frame count: missing camera directory; viewer will use MP4 fallback")

    viewer = RecordingViewer(
        recording_dir=recording_dir,
        frame_data=frames,
        playback_fps=playback_fps,
        image_width=args.image_width,
        mp4_counts=mp4_counts,
        joint_names=joint_names,
        joint_values=joint_values,
        task_description=task_description,
        font_family_override=args.font_family,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
