"""Simple MP4 recording helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MultiViewVideoRecorder:
    def __init__(self, output_dir: str | Path, fps: int, enabled: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.enabled = enabled
        self.frames: dict[str, list[np.ndarray]] = {}

    def add(self, view_name: str, frame: np.ndarray | None) -> None:
        if not self.enabled or frame is None:
            return
        frame = np.asarray(frame)
        if frame.ndim != 3:
            return
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        self.frames.setdefault(view_name, []).append(frame.astype(np.uint8))

    def close(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        outputs: dict[str, str] = {}
        for view_name, frames in self.frames.items():
            if not frames:
                continue
            path = self.output_dir / f"{view_name}.mp4"
            _write_mp4(path, frames, self.fps)
            outputs[view_name] = str(path)
        return outputs


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        return
    except Exception:
        pass

    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write mp4 video {path}") from exc

