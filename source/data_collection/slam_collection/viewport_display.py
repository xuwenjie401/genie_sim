"""Viewport helpers for interactive Galbot SLAM teleop."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.base_utils.logger import logger
from slam_collection.base_motion import yaw_from_quat_wxyz


@dataclass
class ThirdPersonViewConfig:
    enabled: bool = True
    camera_prim: str = "/World/SLAMThirdPersonCamera"
    relative_position: tuple[float, float, float] = (-2.6, -2.2, 1.6)
    look_at: tuple[float, float, float] = (0.45, 0.0, 0.75)
    focal_length: float = 22.0


@dataclass
class HeadViewConfig:
    enabled: bool = True
    camera_prim: str = ""
    window_title: str = "Galbot Head Camera"
    width: int = 640
    height: int = 480


def clear_viewport_selection() -> None:
    """Clear selected prims so viewport transform gizmos are not visible."""

    try:
        import omni.usd

        selection = omni.usd.get_context().get_selection()
        if selection is not None:
            selection.clear_selected_prim_paths()
    except Exception as exc:
        logger.warning(f"Failed to clear viewport selection: {exc}")


def remove_known_debug_prims() -> None:
    """Remove helper prims from older interactive viewport experiments if present."""

    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for prim_path in (
            "/World/SLAMViewportTarget",
            "/World/SLAMViewportAxes",
            "/World/SLAMViewportSphere",
        ):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                stage.RemovePrim(prim_path)
    except Exception as exc:
        logger.warning(f"Failed to remove viewport debug prims: {exc}")


def third_person_view_from_config(task_info: dict) -> ThirdPersonViewConfig:
    viewport_setting = task_info.get("slam_setting", {}).get("viewport", {})
    third_person = viewport_setting.get("third_person", {})
    return ThirdPersonViewConfig(
        enabled=bool(third_person.get("enabled", True)),
        camera_prim=str(third_person.get("camera_prim", ThirdPersonViewConfig.camera_prim)),
        relative_position=_tuple3(
            third_person.get("relative_position"),
            ThirdPersonViewConfig.relative_position,
        ),
        look_at=_tuple3(third_person.get("look_at"), ThirdPersonViewConfig.look_at),
        focal_length=float(third_person.get("focal_length", ThirdPersonViewConfig.focal_length)),
    )


def head_view_from_config(task_info: dict) -> HeadViewConfig:
    recording_setting = task_info.get("recording_setting", {})
    camera_list = recording_setting.get("camera_list", [])
    default_head_camera = camera_list[0] if camera_list else ""

    viewport_setting = task_info.get("slam_setting", {}).get("viewport", {})
    head_view = viewport_setting.get("head_camera", {})
    return HeadViewConfig(
        enabled=bool(head_view.get("enabled", True)),
        camera_prim=str(head_view.get("camera_prim", default_head_camera)),
        window_title=str(head_view.get("window_title", HeadViewConfig.window_title)),
        width=int(head_view.get("width", HeadViewConfig.width)),
        height=int(head_view.get("height", HeadViewConfig.height)),
    )


class ViewportDisplay:
    """Keep the main viewport behind the robot and optionally show the head camera."""

    def __init__(
        self,
        third_person_config: ThirdPersonViewConfig,
        head_view_config: HeadViewConfig,
    ) -> None:
        self.third_person_config = third_person_config
        self.head_view_config = head_view_config
        self._main_viewport = None
        self._head_window = None
        self._active = False

    def initialize(self, position, orientation) -> None:
        clear_viewport_selection()
        remove_known_debug_prims()

        if self.third_person_config.enabled:
            self._initialize_third_person_camera()
            self.update(position, orientation)

        if self.head_view_config.enabled:
            self._initialize_head_view()

        clear_viewport_selection()

    def update(self, position, orientation) -> None:
        if not self.third_person_config.enabled:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            eye, target = self._third_person_eye_target(position, orientation)
            set_camera_view(
                eye=[float(eye[0]), float(eye[1]), float(eye[2])],
                target=[float(target[0]), float(target[1]), float(target[2])],
                camera_prim_path=self.third_person_config.camera_prim,
            )
            if self._main_viewport is not None and self._active:
                self._set_viewport_camera(self._main_viewport, self.third_person_config.camera_prim)
        except Exception as exc:
            logger.warning(f"Failed to update third-person viewport camera: {exc}")

    def _initialize_third_person_camera(self) -> None:
        try:
            import omni.usd
            from omni.kit.viewport.utility import get_active_viewport_and_window
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            camera = UsdGeom.Camera.Define(stage, self.third_person_config.camera_prim)
            camera.CreateFocalLengthAttr().Set(float(self.third_person_config.focal_length))
            viewport, _window = get_active_viewport_and_window()
            self._main_viewport = viewport
            if viewport is not None:
                self._set_viewport_camera(viewport, self.third_person_config.camera_prim)
                self._active = True
            else:
                logger.warning("No active viewport available for third-person follow camera")
        except Exception as exc:
            logger.warning(f"Failed to initialize third-person viewport camera: {exc}")

    def _initialize_head_view(self) -> None:
        camera_prim = self.head_view_config.camera_prim
        if not camera_prim:
            logger.warning("Head camera viewport is enabled, but no head camera prim was configured")
            return

        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(camera_prim)
            if prim is None or not prim.IsValid():
                logger.warning(f"Head camera prim does not exist: {camera_prim}")
                return

            self._head_window = _create_viewport_window(
                self.head_view_config.window_title,
                int(self.head_view_config.width),
                int(self.head_view_config.height),
            )
            viewport = getattr(self._head_window, "viewport_api", self._head_window)
            self._set_viewport_camera(viewport, camera_prim)
            logger.info(f"Head camera viewport opened: {camera_prim}")
        except Exception as exc:
            logger.warning(
                "Failed to open separate head camera viewport. "
                f"Main teleop still works; camera prim: {camera_prim}; error: {exc}"
            )

    def _third_person_eye_target(self, position, orientation) -> tuple[np.ndarray, np.ndarray]:
        base_position = np.array(position, dtype=np.float64)
        yaw = yaw_from_quat_wxyz(orientation)
        rotation = _yaw_rotation_matrix(yaw)
        relative_position = np.array(self.third_person_config.relative_position, dtype=np.float64)
        look_at = np.array(self.third_person_config.look_at, dtype=np.float64)
        return base_position + rotation @ relative_position, base_position + rotation @ look_at

    @staticmethod
    def _set_viewport_camera(viewport, camera_prim_path: str) -> None:
        if hasattr(viewport, "set_active_camera"):
            viewport.set_active_camera(camera_prim_path)
        elif hasattr(viewport, "camera_path"):
            viewport.camera_path = camera_prim_path
        else:
            raise RuntimeError("Viewport object has no supported camera setter")


def _yaw_rotation_matrix(yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _tuple3(value, default) -> tuple[float, float, float]:
    if value is None:
        return tuple(float(item) for item in default)
    if len(value) != 3:
        raise ValueError("viewport vector settings must contain exactly 3 values")
    return tuple(float(item) for item in value)


def _create_viewport_window(title: str, width: int, height: int):
    try:
        from omni.kit.viewport.utility import create_viewport_window

        return create_viewport_window(title, width=width, height=height)
    except Exception:
        from omni.kit.viewport.window import ViewportWindow

        return ViewportWindow(title, width=width, height=height)
