"""ROS2 bag recording helpers for single-process SLAM teleop."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from common.base_utils.logger import logger


DEFAULT_BAG_OUTPUT_DIR = "/home/agxi/Datasets/output/galbot_home"
DEFAULT_BAG_MAX_CACHE_SIZE = 1024 * 1024 * 1024


class SlamRosbagRecorder:
    """Manage Isaac ROS publishers, robot masks, and one ros2 bag process."""

    def __init__(
        self,
        *,
        task_info: dict[str, Any],
        task_config_path: str,
        repo_root: str,
        robot_cfg,
        scene_usd: str,
        scene_usd_path: str,
        robot_usd_path: str,
        rendering_dt: float,
    ) -> None:
        self.task_info = task_info
        self.task_config_path = task_config_path
        self.repo_root = Path(repo_root)
        self.robot_cfg = robot_cfg
        self.scene_usd = scene_usd
        self.scene_usd_path = scene_usd_path
        self.robot_usd_path = robot_usd_path
        self.rendering_dt = float(rendering_dt)

        recording_setting = task_info.get("recording_setting", {})
        self.task_name = str(task_info.get("task", "galbot_slam"))
        self.fps = int(recording_setting.get("fps", 30))
        self.ros_domain_id = int(recording_setting.get("ros_domain_id", 0))
        self.bag_output_dir = self._resolve_output_dir(
            str(recording_setting.get("bag_output_dir", DEFAULT_BAG_OUTPUT_DIR))
        )
        self.prewarm_publishers = bool(recording_setting.get("prewarm_publishers", False))
        self.bag_max_cache_size = int(recording_setting.get("bag_max_cache_size", DEFAULT_BAG_MAX_CACHE_SIZE))
        self.bag_storage_preset_profile = str(
            recording_setting.get("bag_storage_preset_profile", "none")
        ).strip()
        self.bag_use_sim_time = bool(recording_setting.get("bag_use_sim_time", False))
        self.mask_annotator = str(recording_setting.get("mask_annotator", "instance_id_segmentation")).strip()
        self.mask_robot_prim_path = str(
            recording_setting.get("mask_robot_prim_path", self.robot_cfg.robot_prim_path)
        ).rstrip("/")
        self.mask_semantic_filter = str(recording_setting.get("mask_semantic_filter", "class:robot")).strip()
        self.mask_fallback_to_nonzero = bool(recording_setting.get("mask_fallback_to_nonzero_semantic", False))
        self.camera_list = _string_list(recording_setting.get("camera_list"))
        self.depth_camera_list = set(_string_list(recording_setting.get("depth_camera_list")))
        self.mask_camera_list = set(_string_list(recording_setting.get("mask_camera_list")))
        self.tf_targets = self._build_tf_targets(recording_setting)
        self.camera_topics = self._build_camera_topics()
        self.topic_list = self._build_topic_list()

        self.render_step_size = self._render_step_size()
        self._sensor_base = None
        self._camera_resources: list[dict[str, Any]] = []
        self._camera_info_publishers: list[_CameraInfoPublisher] = []
        self._camera_intrinsics: dict[str, dict[str, Any]] = {}
        self._mask_publishers: list[_RobotMaskPublisher] = []
        self._rclpy_node = None
        self._owns_rclpy = False
        self._publishers_initialized = False

        self._process: subprocess.Popen | None = None
        self._recording_path: Path | None = None
        self._recording_started_wall: str | None = None
        self._recording_stopped_wall: str | None = None
        self._last_recording_ready = False

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def recording_path(self) -> Path | None:
        return self._recording_path

    def initialize(self) -> None:
        self._ensure_publishers_initialized()

    def start(self) -> None:
        if self.is_recording:
            logger.warning(f"SLAM rosbag recording is already running: {self._recording_path}")
            return
        if self._process is not None and self._process.poll() is None:
            logger.warning("Found an active rosbag process while starting; ignoring duplicate start")
            return

        self._ensure_publishers_initialized()
        self._recording_path = self._next_recording_path()
        self._recording_started_wall = _wall_time_string()
        self._recording_stopped_wall = None
        self._last_recording_ready = False

        ros_distro = os.getenv("ROS_CMD_DISTRO", "humble")
        record_args = ["ros2", "bag", "record", "-o", str(self._recording_path)]
        if self.bag_max_cache_size > 0:
            record_args.extend(["--max-cache-size", str(self.bag_max_cache_size)])
        if self.bag_storage_preset_profile:
            record_args.extend(["--storage-preset-profile", self.bag_storage_preset_profile])
        if self.bag_use_sim_time:
            record_args.append("--use-sim-time")
        record_args.extend(self.topic_list)
        command = f"""
unset PYTHONPATH
unset LD_LIBRARY_PATH
source /opt/ros/{shlex.quote(ros_distro)}/setup.bash
exec {shlex.join(record_args)}
"""
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        logger.info(f"Starting SLAM rosbag recording: {self._recording_path}")
        logger.info(f"SLAM rosbag topics: {', '.join(self.topic_list)}")
        self._process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            preexec_fn=os.setsid,
            env=env,
        )

    def stop(self) -> None:
        if self._process is None:
            logger.warning("SLAM rosbag recording is not running; stop ignored")
            return

        process = self._process
        recording_path = self._recording_path
        self._recording_stopped_wall = _wall_time_string()
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                logger.info(f"Sent SIGINT to SLAM rosbag process group {process.pid}")
            except ProcessLookupError:
                logger.info(f"SLAM rosbag process {process.pid} has already exited")
            except Exception as exc:
                logger.warning(f"Failed to signal SLAM rosbag process {process.pid}: {exc}")

        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            logger.error(f"SLAM rosbag process {process.pid} did not stop after SIGINT; forcing SIGKILL")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5.0)
            except Exception as exc:
                logger.error(f"Failed to force-stop SLAM rosbag process {process.pid}: {exc}")

        ready = False
        if recording_path is not None:
            ready = self._wait_for_recording_metadata(recording_path, timeout_sec=10.0)
            self._write_recording_info(recording_path, ready)
        self._last_recording_ready = ready
        self._process = None
        if ready:
            logger.info(f"Stopped SLAM rosbag recording: {recording_path}")
        else:
            logger.error(f"Stopped SLAM rosbag recording, but metadata is incomplete: {recording_path}")

    def tick(self, current_time: float) -> None:
        if not self.is_recording:
            return
        for camera_info_publisher in self._camera_info_publishers:
            camera_info_publisher.tick(current_time)
        for mask_publisher in self._mask_publishers:
            mask_publisher.tick(current_time)
        if self._rclpy_node is not None:
            try:
                import rclpy

                rclpy.spin_once(self._rclpy_node, timeout_sec=0.0)
            except Exception as exc:
                logger.warning(f"Failed to spin SLAM mask publisher node: {exc}")

    def shutdown(self) -> None:
        if self.is_recording or self._process is not None:
            self.stop()
        for mask_publisher in self._mask_publishers:
            mask_publisher.destroy()
        self._mask_publishers = []
        for camera_info_publisher in self._camera_info_publishers:
            camera_info_publisher.destroy()
        self._camera_info_publishers = []

        if self._sensor_base is not None and self._camera_resources:
            try:
                self._sensor_base.cleanup_camera_resources(self._camera_resources)
            except Exception as exc:
                logger.warning(f"Failed to cleanup SLAM camera resources: {exc}")
        self._camera_resources = []

        if self._rclpy_node is not None:
            try:
                self._rclpy_node.destroy_node()
            except Exception:
                pass
            self._rclpy_node = None
        if self._owns_rclpy:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self._owns_rclpy = False

    def _ensure_publishers_initialized(self) -> None:
        if self._publishers_initialized:
            return

        self._validate_camera_config()
        os.environ["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        self._init_rclpy_node()

        from isaacsim.sensors.camera import Camera
        from server.ros_publisher.base import USDBase
        from server.ros_publisher.camera import publish_depth, publish_rgb

        self._sensor_base = USDBase()
        self._sensor_base._init_sensor(self.ros_domain_id)
        self._sensor_base.publish_clock()
        self._sensor_base.publish_tf(
            robot_prim=self.robot_cfg.robot_prim_path,
            targets=self.tf_targets,
            approx_freq=1,
            delta_time=self.rendering_dt,
        )

        for camera_prim in self.camera_list:
            width, height = self._camera_resolution(camera_prim)
            camera = Camera(
                prim_path=camera_prim,
                frequency=self.fps,
                resolution=(width, height),
            )
            camera.initialize()
            self._camera_resources.append(publish_rgb(camera, self.render_step_size))
            camera_info = _camera_info_from_prim(camera_prim, width, height)
            self._camera_intrinsics[camera_prim] = camera_info
            self._camera_info_publishers.append(
                _CameraInfoPublisher(
                    node=self._rclpy_node,
                    topic=self.camera_topics[camera_prim]["camera_info"],
                    frame_id=_frame_id(camera_prim),
                    camera_info=camera_info,
                    render_step_size=self.render_step_size,
                )
            )
            if camera_prim in self.depth_camera_list:
                self._camera_resources.append(publish_depth(camera, self.render_step_size))
            if camera_prim in self.mask_camera_list:
                self._mask_publishers.append(
                    _RobotMaskPublisher(
                        node=self._rclpy_node,
                        camera_prim=camera_prim,
                        render_product_path=camera._render_product_path,
                        frame_id=_frame_id(camera_prim),
                        topic=self.camera_topics[camera_prim]["mask"],
                        width=width,
                        height=height,
                        render_step_size=self.render_step_size,
                        annotator_name=self.mask_annotator,
                        robot_prim_path=self.mask_robot_prim_path,
                        semantic_filter=self.mask_semantic_filter,
                        fallback_to_nonzero=self.mask_fallback_to_nonzero,
                    )
                )

        self._publishers_initialized = True
        logger.info(
            f"Initialized SLAM ROS publishers; image topics target {self.fps} Hz "
            f"with render step {self.render_step_size}"
        )

    def _init_rclpy_node(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.parameter import Parameter

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._rclpy_node = Node(
            "slam_robot_mask_publisher",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

    def _validate_camera_config(self) -> None:
        if not self.camera_list:
            raise ValueError("recording_setting.camera_list must include at least one camera")
        unknown_depth = self.depth_camera_list.difference(self.camera_list)
        unknown_mask = self.mask_camera_list.difference(self.camera_list)
        if unknown_depth:
            raise ValueError(f"depth_camera_list contains cameras not in camera_list: {sorted(unknown_depth)}")
        if unknown_mask:
            raise ValueError(f"mask_camera_list contains cameras not in camera_list: {sorted(unknown_mask)}")

    def _resolve_output_dir(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.repo_root / path
        return path

    def _next_recording_path(self) -> Path:
        self.bag_output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        candidate = self.bag_output_dir / stem
        suffix = 1
        while candidate.exists():
            candidate = self.bag_output_dir / f"{stem}_{suffix:02d}"
            suffix += 1
        return candidate

    def _render_step_size(self) -> int:
        if self.fps <= 0:
            return 1
        if self.rendering_dt <= 0.0:
            return 1
        return max(1, int(round(1.0 / (self.rendering_dt * float(self.fps)))))

    def _camera_resolution(self, camera_prim: str) -> tuple[int, int]:
        resolution = getattr(self.robot_cfg, "cameras", {}).get(camera_prim, [640, 480])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            return (640, 480)
        return (int(resolution[0]), int(resolution[1]))

    def _build_tf_targets(self, recording_setting: dict[str, Any]) -> list[str]:
        configured = _string_list(recording_setting.get("tf_targets"))
        if configured:
            targets = configured
        else:
            robot_root = self.robot_cfg.robot_prim_path
            targets = [
                robot_root,
                f"{robot_root}/base_link",
                f"{robot_root}/torso_base_link",
                *self.camera_list,
            ]
        return _dedupe(targets)

    def _build_camera_topics(self) -> dict[str, dict[str, str]]:
        topics = {}
        for camera_prim in self.camera_list:
            frame_id = _frame_id(camera_prim)
            topics[camera_prim] = {
                "rgb": f"/{frame_id}_rgb",
                "depth": f"/{frame_id}_depth" if camera_prim in self.depth_camera_list else "",
                "camera_info": f"/{frame_id}_camera_info",
                "mask": f"/{frame_id}_robot_mask" if camera_prim in self.mask_camera_list else "",
            }
        return topics

    def _build_topic_list(self) -> list[str]:
        topics = ["/clock", "/tf"]
        for camera_prim in self.camera_list:
            camera_topics = self.camera_topics[camera_prim]
            topics.append(camera_topics["rgb"])
            topics.append(camera_topics["camera_info"])
            if camera_topics["depth"]:
                topics.append(camera_topics["depth"])
            if camera_topics["mask"]:
                topics.append(camera_topics["mask"])
        return _dedupe(topics)

    def _wait_for_recording_metadata(self, recording_path: Path, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if (recording_path / "metadata.yaml").is_file() and _has_bag_payload(recording_path):
                return True
            time.sleep(0.1)
        return (recording_path / "metadata.yaml").is_file() and _has_bag_payload(recording_path)

    def _write_recording_info(self, recording_path: Path, recording_ready: bool) -> None:
        recording_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": self.task_name,
            "task_config_path": self.task_config_path,
            "recording_path": str(recording_path),
            "recording_ready": bool(recording_ready),
            "started_wall_time": self._recording_started_wall,
            "stopped_wall_time": self._recording_stopped_wall,
            "fps": self.fps,
            "image_topics_fps": self.fps,
            "image_publish_step_size": self.render_step_size,
            "ros_domain_id": self.ros_domain_id,
            "ros_distro": os.getenv("ROS_CMD_DISTRO", "humble"),
            "bag_max_cache_size": self.bag_max_cache_size,
            "bag_storage_preset_profile": self.bag_storage_preset_profile,
            "bag_use_sim_time": self.bag_use_sim_time,
            "robot": {
                "name": self.robot_cfg.robot_name,
                "prim_path": self.robot_cfg.robot_prim_path,
                "usd_path": self.robot_usd_path,
                "init_pose": self.task_info.get("robot", {}).get("robot_init_pose", {}),
            },
            "scene": {
                "scene_usd": self.scene_usd,
                "scene_usd_path": self.scene_usd_path,
                "scene_id": self.task_info.get("scene", {}).get("scene_id", ""),
            },
            "topics": self.topic_list,
            "camera_topics": self.camera_topics,
            "camera_intrinsics": self._camera_intrinsics,
            "camera_list": self.camera_list,
            "depth_camera_list": sorted(self.depth_camera_list),
            "mask_camera_list": sorted(self.mask_camera_list),
            "tf_targets": self.tf_targets,
            "mask_method": {
                "type": "semantic_segmentation",
                "semantic_label": "robot",
                "annotator": self.mask_annotator,
                "robot_prim_path": self.mask_robot_prim_path,
                "semantic_filter": self.mask_semantic_filter,
                "fallback_to_nonzero_semantic": self.mask_fallback_to_nonzero,
                "encoding": "mono8",
                "robot_pixel": 255,
                "background_pixel": 0,
            },
        }
        info_path = recording_path / "recording_info.json"
        with info_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
            file.write("\n")


class _CameraInfoPublisher:
    def __init__(
        self,
        *,
        node,
        topic: str,
        frame_id: str,
        camera_info: dict[str, Any],
        render_step_size: int,
    ) -> None:
        from sensor_msgs.msg import CameraInfo

        self.node = node
        self.topic = topic
        self.frame_id = frame_id
        self.camera_info = camera_info
        self.render_step_size = max(1, int(render_step_size))
        self._step_count = 0
        self._message_type = CameraInfo
        self.publisher = node.create_publisher(CameraInfo, topic, 10)
        logger.info(
            f"Initialized camera info publisher {topic}: "
            f"fx={camera_info['k'][0][0]:.3f}, fy={camera_info['k'][1][1]:.3f}, "
            f"cx={camera_info['k'][0][2]:.3f}, cy={camera_info['k'][1][2]:.3f}"
        )

    def tick(self, current_time: float) -> None:
        self._step_count += 1
        if self._step_count < self.render_step_size:
            return
        self._step_count = 0

        msg = self._message_type()
        sec = int(current_time)
        nanosec = int((float(current_time) - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.frame_id
        msg.width = int(self.camera_info["width"])
        msg.height = int(self.camera_info["height"])
        msg.distortion_model = str(self.camera_info["distortion_model"])
        msg.d = list(self.camera_info["d"])
        msg.k = _flatten_matrix(self.camera_info["k"])
        msg.r = _flatten_matrix(self.camera_info["r"])
        msg.p = _flatten_matrix(self.camera_info["p"])
        self.publisher.publish(msg)

    def destroy(self) -> None:
        try:
            self.node.destroy_publisher(self.publisher)
        except Exception:
            pass


class _RobotMaskPublisher:
    def __init__(
        self,
        *,
        node,
        camera_prim: str,
        render_product_path: str,
        frame_id: str,
        topic: str,
        width: int,
        height: int,
        render_step_size: int,
        annotator_name: str,
        robot_prim_path: str,
        semantic_filter: str,
        fallback_to_nonzero: bool,
    ) -> None:
        import omni.replicator.core as rep
        from sensor_msgs.msg import Image

        self.node = node
        self.camera_prim = camera_prim
        self.render_product_path = render_product_path
        self.frame_id = frame_id
        self.topic = topic
        self.width = int(width)
        self.height = int(height)
        self.render_step_size = max(1, int(render_step_size))
        self.annotator_name = annotator_name or "instance_id_segmentation"
        self.robot_prim_path = robot_prim_path.rstrip("/")
        self.semantic_filter = semantic_filter
        self.fallback_to_nonzero = fallback_to_nonzero
        self._step_count = 0
        self._warned_no_robot_label = False
        self._warned_bad_data = False
        self._warned_nonzero_fallback = False
        self._image_type = Image
        self.publisher = node.create_publisher(Image, topic, 10)
        init_params = {"colorize": False}
        if self.annotator_name == "semantic_segmentation" and semantic_filter:
            init_params["semanticFilter"] = semantic_filter
        elif self.annotator_name == "semantic_segmentation":
            init_params["semanticTypes"] = ["class"]
        self.annotator = rep.AnnotatorRegistry.get_annotator(
            self.annotator_name,
            init_params=init_params,
        )
        _attach_annotator(self.annotator, render_product_path)
        logger.info(
            f"Initialized robot mask publisher {topic} from {camera_prim}, "
            f"annotator={self.annotator_name}, robot_prim={self.robot_prim_path}"
        )

    def tick(self, current_time: float) -> None:
        self._step_count += 1
        if self._step_count < self.render_step_size:
            return
        self._step_count = 0
        mask = self._read_mask()
        if mask is None:
            return

        msg = self._image_type()
        sec = int(current_time)
        nanosec = int((float(current_time) - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = self.width
        msg.data = np.ascontiguousarray(mask).tobytes()
        self.publisher.publish(msg)

    def destroy(self) -> None:
        try:
            self.annotator.detach([self.render_product_path])
        except Exception:
            try:
                self.annotator.detach(self.render_product_path)
            except Exception:
                pass
        try:
            self.node.destroy_publisher(self.publisher)
        except Exception:
            pass

    def _read_mask(self) -> np.ndarray | None:
        try:
            raw = self.annotator.get_data()
        except Exception as exc:
            if not self._warned_bad_data:
                logger.warning(f"Failed to read robot mask annotator for {self.camera_prim}: {exc}")
                self._warned_bad_data = True
            return None

        data = raw.get("data") if isinstance(raw, dict) else raw
        info = raw.get("info", {}) if isinstance(raw, dict) else {}
        if data is None:
            return None
        segmentation = np.asarray(data)
        if segmentation.ndim == 3:
            segmentation = segmentation[:, :, 0]
        if segmentation.ndim != 2:
            if not self._warned_bad_data:
                logger.warning(f"Unexpected robot mask shape for {self.camera_prim}: {segmentation.shape}")
                self._warned_bad_data = True
            return None

        if self.annotator_name.startswith("instance"):
            robot_ids = _robot_path_ids(info.get("idToLabels", {}), self.robot_prim_path)
        else:
            robot_ids = _robot_label_ids(info.get("idToLabels", {}))
        if not robot_ids:
            if self.fallback_to_nonzero:
                mask = (segmentation != 0).astype(np.uint8) * 255
                if mask.shape != (self.height, self.width):
                    mask = _resize_nearest(mask, self.width, self.height)
                if np.any(mask):
                    if not self._warned_nonzero_fallback:
                        logger.warning(
                            f"No explicit robot semantic id for {self.topic}; using nonzero semantic fallback"
                        )
                        self._warned_nonzero_fallback = True
                    return mask
            if not self._warned_no_robot_label:
                logger.warning(
                    f"No robot id visible yet for mask topic {self.topic}; "
                    f"annotator={self.annotator_name}, robot_prim={self.robot_prim_path}"
                )
                self._warned_no_robot_label = True
            return np.zeros((self.height, self.width), dtype=np.uint8)

        mask = np.isin(segmentation, list(robot_ids)).astype(np.uint8) * 255
        if mask.shape != (self.height, self.width):
            mask = _resize_nearest(mask, self.width, self.height)
        return mask


def _attach_annotator(annotator, render_product_path: str) -> None:
    try:
        annotator.attach([render_product_path])
    except Exception:
        annotator.attach(render_product_path)


def _camera_info_from_prim(camera_prim: str, width: int, height: int) -> dict[str, Any]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim)
    if not prim or not prim.IsValid():
        raise ValueError(f"Invalid camera prim for CameraInfo: {camera_prim}")

    width = int(width)
    height = int(height)
    focal_length = _attr_float(prim, "focalLength", 1.0)
    horizontal_aperture = _attr_float(prim, "horizontalAperture", 1.0)
    vertical_aperture = _attr_float(prim, "verticalAperture", 1.0)
    horizontal_offset = _attr_float(prim, "horizontalApertureOffset", 0.0)
    vertical_offset = _attr_float(prim, "verticalApertureOffset", 0.0)
    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    # In rendered image coordinates, USD aperture offsets shift the principal point
    # opposite the authored filmback offset direction.
    cx = width * 0.5 - horizontal_offset * width / horizontal_aperture
    cy = height * 0.5 - vertical_offset * height / vertical_aperture
    source = "usd_camera_focal_aperture"
    d = [0.0, 0.0, 0.0, 0.0, 0.0]
    k = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]
    r = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    p = [
        [float(fx), 0.0, float(cx), 0.0],
        [0.0, float(fy), float(cy), 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return {
        "width": width,
        "height": height,
        "distortion_model": "plumb_bob",
        "d": [float(value) for value in d],
        "k": k,
        "r": r,
        "p": p,
        "source": source,
        "aperture_offset_convention": "principal_point = image_center - aperture_offset * image_size / aperture",
        "focal_aperture": {
            "focalLength": float(focal_length),
            "horizontalAperture": float(horizontal_aperture),
            "verticalAperture": float(vertical_aperture),
            "horizontalApertureOffset": float(horizontal_offset),
            "verticalApertureOffset": float(vertical_offset),
        },
        "opencv_pinhole_usd_attrs": {
            "fx": _attr_float(prim, "omni:lensdistortion:opencvPinhole:fx"),
            "fy": _attr_float(prim, "omni:lensdistortion:opencvPinhole:fy"),
            "cx": _attr_float(prim, "omni:lensdistortion:opencvPinhole:cx"),
            "cy": _attr_float(prim, "omni:lensdistortion:opencvPinhole:cy"),
            "k1": _attr_float(prim, "omni:lensdistortion:opencvPinhole:k1", 0.0),
            "k2": _attr_float(prim, "omni:lensdistortion:opencvPinhole:k2", 0.0),
            "p1": _attr_float(prim, "omni:lensdistortion:opencvPinhole:p1", 0.0),
            "p2": _attr_float(prim, "omni:lensdistortion:opencvPinhole:p2", 0.0),
            "k3": _attr_float(prim, "omni:lensdistortion:opencvPinhole:k3", 0.0),
            "imageSize": _attr_value(prim, "omni:lensdistortion:opencvPinhole:imageSize"),
        },
    }


def _attr_value(prim, attr_name: str):
    attr = prim.GetAttribute(attr_name)
    if not attr:
        return None
    return attr.Get()


def _attr_float(prim, attr_name: str, default: float | None = None) -> float | None:
    value = _attr_value(prim, attr_name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flatten_matrix(matrix: Any) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _resize_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return mask[:height, :width]


def _robot_label_ids(id_to_labels: Any) -> set[int]:
    robot_ids: set[int] = set()
    if not isinstance(id_to_labels, dict):
        return robot_ids
    for key, value in id_to_labels.items():
        if _contains_robot_label(value):
            try:
                robot_ids.add(int(key))
            except (TypeError, ValueError):
                continue
    return robot_ids


def _contains_robot_label(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_robot_label(inner) for inner in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_robot_label(inner) for inner in value)
    text = str(value).strip().lower()
    return text == "robot" or "robot" in {token.strip() for token in text.replace(";", ",").split(",")}


def _robot_path_ids(id_to_labels: Any, robot_prim_path: str) -> set[int]:
    robot_ids: set[int] = set()
    if not isinstance(id_to_labels, dict):
        return robot_ids
    root = robot_prim_path.rstrip("/")
    for key, value in id_to_labels.items():
        if _contains_robot_path(value, root):
            try:
                robot_ids.add(int(key))
            except (TypeError, ValueError):
                continue
    return robot_ids


def _contains_robot_path(value: Any, robot_prim_path: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_robot_path(inner, robot_prim_path) for inner in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_robot_path(inner, robot_prim_path) for inner in value)
    text = str(value).strip()
    return text == robot_prim_path or text.startswith(f"{robot_prim_path}/")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _frame_id(camera_prim: str) -> str:
    return camera_prim.rstrip("/").split("/")[-1]


def _has_bag_payload(recording_path: Path) -> bool:
    if not recording_path.is_dir():
        return False
    return any(recording_path.glob("*.db3")) or any(recording_path.glob("*.mcap"))


def _wall_time_string() -> str:
    return datetime.now().isoformat(timespec="seconds")
