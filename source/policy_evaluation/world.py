"""Direct Isaac resource management for policy evaluation.

This module intentionally does not use the data-collection gRPC server,
ROS publishers, or rosbag recording pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

import numpy as np

from policy_evaluation.config import AdapterConfig, CameraConfig


class EvalWorld:
    def __init__(
        self,
        physics_step: int,
        rendering_step: int,
        render: bool = True,
        device: str = "cpu",
        config: Any | None = None,
    ):
        self.physics_step = physics_step
        self.rendering_step = rendering_step
        self.render = render
        self.device = device
        self.config = config
        assets_root = os.environ.get("SIM_ASSETS")
        if not assets_root:
            raise RuntimeError("SIM_ASSETS must point to GenieSimAssets.")
        self.assets_root = Path(assets_root)
        if not self.assets_root.exists():
            raise RuntimeError(f"SIM_ASSETS does not exist: {self.assets_root}")
        self.world = None
        self.stage = None
        self.robot = None
        self.robot_cfg: dict[str, Any] = {}
        self.robot_prim_path = ""
        self.object_prim_paths: dict[str, str] = {}
        self.rigid_bodies: dict[str, Any] = {}
        self.cameras: dict[str, Any] = {}
        self.camera_resolutions: dict[str, list[int]] = {}
        self.gripper_control: dict[str, Any] = self._default_gripper_control()
        self.gripper_drive_settings: dict[str, dict[str, float]] = {}
        self.posture_hold_targets: dict[str, float] = {}
        self.target_object_id = ""
        self.place_target_object_id = ""

    def setup_episode(self, task_instance: dict[str, Any], adapter_config: AdapterConfig, camera_config: CameraConfig):
        self._new_stage()
        self._configure_viewport_perspective(camera_config.observer)
        self._load_scene_and_robot(task_instance)
        self._initialize_robot()
        self._resolve_adapter_defaults(adapter_config)
        self._set_initial_joints(task_instance)
        self._initialize_posture_hold_targets()
        self._load_objects(task_instance)
        self._setup_cameras(camera_config)
        self.step_frames(10)

    def step_frames(self, count: int = 1) -> None:
        for _ in range(max(1, count)):
            self._apply_posture_hold()
            self._update_gripper_control()
            self.world.step(render=self.render)

    def step_seconds(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.step_frames(int(round(seconds * self.physics_step)))

    def get_joint_state(self, joint_names: list[str]) -> dict[str, float]:
        indices = self._joint_indices(joint_names)
        positions = self.robot.get_joint_positions(joint_indices=indices)
        return {name: float(value) for name, value in zip(joint_names, positions)}

    def set_joint_positions(self, joint_names: list[str], positions: np.ndarray) -> None:
        indices = self._joint_indices(joint_names)
        positions = np.asarray(positions, dtype=np.float64)
        positions = self._clip_positions(indices, positions)
        self.robot.set_joint_positions(positions, joint_indices=indices)
        self.robot.set_joint_velocities(np.zeros(len(indices), dtype=np.float64), joint_indices=indices)
        if hasattr(self.robot, "set_joint_position_targets"):
            self.robot.set_joint_position_targets(positions, joint_indices=indices)
        if hasattr(self.robot, "set_joint_velocity_targets"):
            self.robot.set_joint_velocity_targets(np.zeros(len(indices), dtype=np.float64), joint_indices=indices)
        self._update_posture_hold_targets(joint_names, positions)

    def command_joint_positions(self, joint_names: list[str], positions: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        indices = self._joint_indices(joint_names)
        positions = np.asarray(positions, dtype=np.float64)
        positions = self._clip_positions(indices, positions)
        self._update_posture_hold_targets(joint_names, positions)
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=positions,
                joint_indices=np.asarray(indices, dtype=np.int32),
            )
        )

    def set_gripper(self, joint_names: list[str], positions: list[float]) -> None:
        if not joint_names or not positions:
            return
        if len(positions) == 1 and len(joint_names) > 1:
            positions = positions * len(joint_names)
        self.set_joint_positions(joint_names, np.asarray(positions[: len(joint_names)], dtype=np.float64))

    def command_gripper(self, arm: str, open_gripper: bool) -> None:
        state = "open" if open_gripper else "close"
        if self.gripper_control.get("arm") == arm and self.gripper_control.get("command") == state:
            return
        self.gripper_control = self._default_gripper_control()
        self.gripper_control.update(
            {
                "arm": arm,
                "command": state,
                "mode": "opening" if open_gripper else "closing",
                "last_sample_position": None,
                "sample_count": 0,
                "frame_count": 0,
                "hold_positions": None,
            }
        )
        if open_gripper:
            self._set_gripper_drive(arm, stiffness=0.0, max_force=100.0)
        else:
            self._set_gripper_drive(
                arm,
                stiffness=0.0,
                max_force=self._config_float("gripper_close_max_force", 0.2),
            )

    def capture(self, prim_path: str) -> np.ndarray:
        if not prim_path:
            raise ValueError("Camera prim_path is empty.")
        if prim_path not in self.cameras:
            self._register_camera(prim_path, self.camera_resolutions.get(prim_path, [640, 480]))
        camera = self.cameras[prim_path]
        rgba = camera.get_rgba()
        if rgba is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        image = np.asarray(rgba)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        return image[..., :3]

    def get_object_position(self, object_id: str) -> np.ndarray:
        pose = self.get_object_pose_matrix(object_id)
        return pose[:3, 3].copy()

    def get_tcp_position(self, arm: str) -> tuple[np.ndarray, str]:
        from isaacsim.core.utils.xforms import get_world_pose

        gripper_cfg = self.robot_cfg.get("gripper", {})
        prim_paths = gripper_cfg.get("end_effector_center_prim_path") or gripper_cfg.get("end_effector_prim_path", {})
        prim_path = prim_paths.get(arm, "") if isinstance(prim_paths, dict) else str(prim_paths or "")
        if not prim_path:
            raise KeyError(f"No TCP prim path configured for arm={arm}")
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise KeyError(f"TCP prim not found for arm={arm}: {prim_path}")
        position, _quaternion = get_world_pose(prim_path)
        return np.asarray(position, dtype=np.float64), prim_path

    def get_object_pose_matrix(self, object_id: str) -> np.ndarray:
        from isaacsim.core.utils.xforms import get_world_pose

        prim_path = self._resolve_object_prim_path(object_id)
        entity_path = prim_path + "/entity"
        prim = self.stage.GetPrimAtPath(entity_path)
        if prim and prim.IsValid():
            position, quaternion = get_world_pose(entity_path)
        else:
            position, quaternion = get_world_pose(prim_path)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quat_wxyz_to_matrix(np.asarray(quaternion, dtype=np.float64))
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    def _new_stage(self) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage

        try:
            World.clear_instance()
        except Exception:
            pass
        create_new_stage()
        self.world = World(
            stage_units_in_meters=1,
            physics_dt=1.0 / self.physics_step,
            rendering_dt=1.0 / self.rendering_step,
            device=self.device,
        )
        self.stage = get_current_stage()
        self.robot = None
        self.object_prim_paths = {}
        self.rigid_bodies = {}
        self.cameras = {}
        self.camera_resolutions = {}
        self.gripper_control = self._default_gripper_control()
        self.gripper_drive_settings = {}
        self.posture_hold_targets = {}
        self.target_object_id = ""
        self.place_target_object_id = ""

    def _load_scene_and_robot(self, task_instance: dict[str, Any]) -> None:
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Sdf, UsdPhysics

        robot_info = task_instance.get("robot", {})
        robot_cfg_name = robot_info.get("robot_cfg")
        if not robot_cfg_name:
            raise ValueError("Generated task instance is missing robot.robot_cfg")
        robot_cfg_path = Path(__file__).resolve().parents[1] / "data_collection" / "config" / "robot_cfg" / robot_cfg_name
        with open(robot_cfg_path, "r", encoding="utf-8") as file:
            self.robot_cfg = json.load(file)

        robot_usd = self.assets_root / self.robot_cfg["robot"]["robot_usd"]
        scene_usd = _resolve_asset_path(self.assets_root, task_instance["scene_usd"])
        self.robot_prim_path = self.robot_cfg["robot"]["base_prim_path"]
        add_reference_to_stage(str(scene_usd), "/World")
        add_reference_to_stage(str(robot_usd), self.robot_prim_path)

        init_pose = robot_info.get("robot_init_pose", {})
        position = init_pose.get("position", [0.0, 0.0, 0.0])
        quaternion = init_pose.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        SingleXFormPrim(prim_path=self.robot_prim_path, position=position, orientation=quaternion)

        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    def _load_objects(self, task_instance: dict[str, Any]) -> None:
        from isaacsim.core.utils.prims import create_prim, get_prim_at_path

        self.target_object_id = _resolve_target_object_id(task_instance)
        self.place_target_object_id = _resolve_place_target_object_id(task_instance)
        if self.target_object_id:
            print(f"[policy_eval] target grasp object={self.target_object_id}", flush=True)
        if self.place_target_object_id:
            print(f"[policy_eval] place target object={self.place_target_object_id}", flush=True)
        objects_prim = get_prim_at_path("/World/Objects")
        if not objects_prim or not objects_prim.IsValid():
            create_prim("/World/Objects", prim_type="Xform")

        required_object_ids = _resolve_required_task_object_ids(task_instance)
        enable_distractors = bool(self._config_value("enable_distractors", False))
        skipped_distractors = 0
        scene_object_poses = []
        for obj in task_instance.get("objects", []):
            object_id = obj.get("object_id")
            if not object_id or "fix_pose" in object_id:
                continue
            if not enable_distractors and required_object_ids and object_id not in required_object_ids:
                skipped_distractors += 1
                continue
            if obj.get("scene_object", False):
                scene_object_poses.append(obj)
                continue
            self._add_or_update_object(obj)
            self.step_seconds(0.1)
            self._add_or_update_object(obj)
            self.step_seconds(0.2)

        if scene_object_poses:
            self._set_scene_object_poses(scene_object_poses)
        if skipped_distractors:
            print(f"[policy_eval] skipped {skipped_distractors} distractor objects", flush=True)
        self.step_seconds(2.0)

    def _add_or_update_object(self, obj: dict[str, Any]) -> None:
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        object_id = obj["object_id"]
        prim_path = obj.get("prim_path", f"/World/Objects/{object_id}")
        prim = self.stage.GetPrimAtPath(prim_path)
        already_in_stage = bool(prim and prim.IsValid())
        if not already_in_stage:
            usd_path = _resolve_object_asset_path(self.assets_root, obj)
            add_reference_to_stage(str(usd_path), prim_path)

        SingleXFormPrim(
            prim_path=prim_path,
            position=obj.get("position", [0.0, 0.0, 0.0]),
            orientation=obj.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
            scale=_resolve_object_scale(obj),
        )

        if not already_in_stage:
            self._apply_object_physics(prim_path, obj)
            print(
                f"[policy_eval] loaded object object_id={object_id} prim={prim_path} "
                f"mass={self._object_mass(obj)} model_type={obj.get('model_type', 'convexDecomposition')} "
                f"static_friction={self._config_float('object_static_friction', obj.get('static_friction', 0.5))} "
                f"dynamic_friction={self._config_float('object_dynamic_friction', obj.get('dynamic_friction', 0.5))}",
                flush=True,
            )
        self.object_prim_paths[object_id] = prim_path

    def _set_scene_object_poses(self, objects: list[dict[str, Any]]) -> None:
        from isaacsim.core.prims import SingleXFormPrim

        for obj in objects:
            object_id = obj["object_id"]
            prim_path = obj.get("prim_path", f"/World/Objects/{object_id}")
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                print(f"[policy_eval] scene object prim not found; skip pose set: {prim_path}", flush=True)
                continue
            SingleXFormPrim(
                prim_path=prim_path,
                position=obj.get("position", [0.0, 0.0, 0.0]),
                orientation=obj.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
                scale=_resolve_object_scale(obj),
            )
            self.object_prim_paths[object_id] = prim_path

    def _apply_object_physics(self, prim_path: str, obj: dict[str, Any]) -> None:
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
        from isaacsim.core.utils.prims import get_prim_object_type
        from omni.physx.scripts import utils
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"Object prim was not created: {prim_path}")

        mesh_paths = [
            str(child.GetPath())
            for child in Usd.PrimRange(prim)
            if child.IsA(UsdGeom.Mesh)
        ]
        static_friction = self._config_float("object_static_friction", obj.get("static_friction", 0.5))
        dynamic_friction = self._config_float("object_dynamic_friction", obj.get("dynamic_friction", 0.5))
        for mesh_path in mesh_paths:
            geometry_prim = SingleGeometryPrim(prim_path=mesh_path, reset_xform_properties=False)
            material_path = f"{mesh_path}/object_physics"
            geometry_prim.apply_physics_material(
                PhysicsMaterial(
                    prim_path=material_path,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=None,
                )
            )
            self._set_friction_combine_max(material_path)

        add_rigid_body = bool(obj.get("add_rigid_body", True))
        if get_prim_object_type(prim_path) == "articulation" or not add_rigid_body:
            return

        object_mass = self._object_mass(obj)
        model_type = obj.get("model_type", "convexDecomposition")
        utils.setRigidBody(prim, model_type, False)
        if bool(self._config_value("disable_object_sleep", True)):
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rb_api.GetSleepThresholdAttr().Set(0.0)
        rigid_prim = SingleRigidPrim(prim_path=prim_path, mass=object_mass)
        mass_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
        mass_api.CreateMassAttr().Set(object_mass)
        self.rigid_bodies[prim_path] = rigid_prim
        rigid_prim.initialize()

    def _initialize_robot(self) -> None:
        from isaacsim.core.prims import SingleArticulation

        self.world.reset()
        self.world.play()
        self.step_frames(5)
        self.robot = SingleArticulation(self.robot_prim_path)
        self.robot.initialize()
        self._configure_robot_drive_strength()
        if (
            self._config_value("gripper_static_friction", None) is not None
            or self._config_value("gripper_dynamic_friction", None) is not None
        ):
            self._apply_gripper_physics_material()
        self.step_frames(2)

    def _resolve_adapter_defaults(self, adapter_config: AdapterConfig) -> None:
        arm = adapter_config.arm
        if not adapter_config.joint_names:
            active = self.robot_cfg["robot"].get("active_arm_joints", {})
            adapter_config.joint_names = list(active.get(arm, []))
        if not adapter_config.gripper_joint_names:
            adapter_config.gripper_joint_names = list(self.robot_cfg["gripper"]["finger_names"].get(arm, []))
        if not adapter_config.open_gripper_positions:
            adapter_config.open_gripper_positions = list(self.robot_cfg["gripper"]["opened_positions"].get(arm, []))
        if not adapter_config.closed_gripper_positions:
            closed_positions = self.robot_cfg.get("closed_positions", {})
            closed_positions = closed_positions or self.robot_cfg["gripper"].get("closed_positions", {})
            adapter_config.closed_gripper_positions = list(closed_positions.get(arm, []))
        if len(adapter_config.joint_names) != adapter_config.action_arm_dim:
            raise ValueError(
                f"Expected {adapter_config.action_arm_dim} arm joints for {arm}, got {adapter_config.joint_names}"
            )

    def _set_initial_joints(self, task_instance: dict[str, Any]) -> None:
        robot_info = task_instance.get("robot", {})
        joint_pose = {}
        joint_pose.update(robot_info.get("fixed_joint_reset_pose", {}))
        joint_pose.update(robot_info.get("init_arm_pose", {}))
        if not joint_pose:
            return
        valid_names = []
        valid_positions = []
        dof_names = set(self.robot.dof_names)
        for name, value in joint_pose.items():
            if name in dof_names and value is not None and np.isfinite(value):
                valid_names.append(name)
                valid_positions.append(float(value))
        if valid_names:
            self.set_joint_positions(valid_names, np.asarray(valid_positions, dtype=np.float64))
            self.step_frames(5)

    def _setup_cameras(self, camera_config: CameraConfig) -> None:
        if not camera_config.head:
            raise ValueError("Evaluation config must set cameras.head.")
        if not camera_config.wrist:
            raise ValueError("Evaluation config must set cameras.wrist.")
        robot_cameras = self.robot_cfg.get("camera", {})
        self.camera_resolutions.update({key: value for key, value in robot_cameras.items()})
        for prim_path in [camera_config.head, camera_config.wrist]:
            if prim_path:
                self._register_camera(prim_path, self.camera_resolutions.get(prim_path, [640, 480]))
        observer = camera_config.observer
        if observer:
            prim_path = observer.get("prim_path", "/World/PolicyEval/ObserverCamera")
            resolution = observer.get("resolution", [640, 480])
            self._register_camera(prim_path, resolution, observer=observer)
            self._configure_viewport_camera(prim_path, observer)

    def _register_camera(self, prim_path: str, resolution: list[int], observer: dict[str, Any] | None = None) -> None:
        from isaacsim.sensors.camera import Camera

        camera = Camera(prim_path=prim_path, resolution=resolution)
        camera.initialize()
        if observer:
            position = _optional_vector(observer.get("position"), expected_len=3)
            target = _optional_vector(observer.get("target"), expected_len=3)
            quaternion = _optional_vector(observer.get("quaternion"), expected_len=4)
            if position is not None and target is not None:
                camera.set_world_pose(
                    position=position,
                    orientation=_look_at_quat_world_axes(position, target),
                    camera_axes="world",
                )
            elif position is not None and quaternion is not None:
                camera.set_world_pose(
                    position=position,
                    orientation=quaternion,
                    camera_axes="usd",
                )
            self._apply_observer_camera_intrinsics(camera, prim_path, observer)
        self.cameras[prim_path] = camera
        self.camera_resolutions[prim_path] = resolution
        self.step_frames(2)

    def _apply_observer_camera_intrinsics(self, camera: Any, prim_path: str, observer: dict[str, Any]) -> None:
        focal_length = float(observer.get("focal_length", 18.14756))
        horizontal_aperture = float(observer.get("horizontal_aperture", 20.955))
        vertical_aperture = float(observer.get("vertical_aperture", 15.2908))
        clipping_range = observer.get("clipping_range", [0.01, 100000.0])

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        prim.GetAttribute("focalLength").Set(focal_length)
        prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
        prim.GetAttribute("verticalAperture").Set(vertical_aperture)
        prim.GetAttribute("clippingRange").Set(tuple(float(value) for value in clipping_range))

    def _configure_viewport_camera(self, prim_path: str, observer: dict[str, Any]) -> None:
        if not self.render or not observer.get("set_viewport", True):
            return
        self._configure_viewport_perspective(observer)

    def _configure_viewport_perspective(self, observer: dict[str, Any] | None) -> None:
        if not observer or not self.render or not observer.get("set_viewport", True):
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport_and_window
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            from pxr import Gf

            position = _optional_vector(observer.get("position"), expected_len=3)
            target = _optional_vector(observer.get("target"), expected_len=3)
            if position is None or target is None:
                return
            camera_state = ViewportCameraState("/OmniverseKit_Persp")
            camera_state.set_position_world(Gf.Vec3d(*position.tolist()), True)
            camera_state.set_target_world(Gf.Vec3d(*target.tolist()), True)

            viewport, _window = get_active_viewport_and_window()
            if viewport is not None:
                viewport.set_active_camera("/OmniverseKit_Persp")
        except Exception as exc:
            print(f"[policy_eval] failed to configure perspective viewport camera: {exc}", flush=True)

    def _resolve_object_prim_path(self, object_id: str) -> str:
        prim_path = self.object_prim_paths.get(object_id)
        if not prim_path:
            prim_path = object_id if object_id.startswith("/") else f"/World/Objects/{object_id}"
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise KeyError(f"Object prim not found for {object_id}: {prim_path}")
        return prim_path

    def _joint_indices(self, joint_names: list[str]) -> list[int]:
        return [self.robot.get_dof_index(name) for name in joint_names]

    def _clip_positions(self, indices: list[int], positions: np.ndarray) -> np.ndarray:
        try:
            lowers = self.robot.dof_properties["lower"][indices]
            uppers = self.robot.dof_properties["upper"][indices]
            return np.clip(positions, lowers, uppers)
        except Exception:
            return positions

    def _config_value(self, name: str, default: Any) -> Any:
        if self.config is None:
            return default
        value = getattr(self.config, name, default)
        return default if value is None else value

    def _config_float(self, name: str, default: Any) -> float:
        return float(self._config_value(name, default))

    def _object_mass(self, obj: dict[str, Any]) -> float:
        object_id = obj.get("object_id", "")
        prim_path = obj.get("prim_path", "")
        overrides = self._config_value("object_mass_overrides", {})
        if isinstance(overrides, dict):
            for key in (object_id, prim_path):
                if key in overrides:
                    return float(overrides[key])
        target_override = self._config_value("target_object_mass_override", None)
        if target_override is not None and object_id == self.target_object_id:
            return float(target_override)
        return float(obj.get("mass", 0.01))

    def _set_friction_combine_max(self, material_path: str) -> None:
        from pxr import PhysxSchema

        material_prim = self.stage.GetPrimAtPath(material_path)
        if not material_prim or not material_prim.IsValid():
            return
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        attr = physx_material_api.GetFrictionCombineModeAttr()
        value = attr.Get()
        if value is None:
            physx_material_api.CreateFrictionCombineModeAttr().Set("max")
        elif value != "max":
            attr.Set("max")

    def _apply_gripper_physics_material(self) -> None:
        try:
            from isaacsim.core.api.materials import PhysicsMaterial
            from isaacsim.core.prims import SingleGeometryPrim
            from pxr import Usd, UsdGeom

            robot_prim = self.stage.GetPrimAtPath(self.robot_prim_path)
            if not robot_prim or not robot_prim.IsValid():
                return
            configured_static = self._config_value("gripper_static_friction", None)
            configured_dynamic = self._config_value("gripper_dynamic_friction", None)
            if configured_static is None and configured_dynamic is None:
                return
            if configured_static is None:
                configured_static = configured_dynamic
            if configured_dynamic is None:
                configured_dynamic = configured_static
            static_friction = float(configured_static)
            dynamic_friction = float(configured_dynamic)
            mesh_count = 0
            for child in Usd.PrimRange(robot_prim):
                path = str(child.GetPath())
                if "gripper" not in path.lower() or not child.IsA(UsdGeom.Mesh):
                    continue
                geometry_prim = SingleGeometryPrim(prim_path=path, reset_xform_properties=False)
                material_path = f"{path}/gripper_physics"
                geometry_prim.apply_physics_material(
                    PhysicsMaterial(
                        prim_path=material_path,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=None,
                    )
                )
                self._set_friction_combine_max(material_path)
                mesh_count += 1
            print(
                f"[policy_eval] applied gripper friction material to {mesh_count} meshes "
                f"static={static_friction} dynamic={dynamic_friction}",
                flush=True,
            )
        except Exception as exc:
            print(f"[policy_eval] failed to apply gripper friction material: {exc}", flush=True)

    def _configure_robot_drive_strength(self) -> None:
        try:
            gripper_names = set(self._all_gripper_joint_names())
            joint_indices = [
                idx
                for idx, name in enumerate(self.robot.dof_names)
                if name not in gripper_names
            ]
            if not joint_indices:
                return
            self.robot._articulation_view.set_max_efforts(
                values=np.full((len(joint_indices),), 5000.0, dtype=np.float64),
                joint_indices=joint_indices,
            )
            self.robot._articulation_view.set_gains(
                kps=np.full((1, len(joint_indices)), 50000.0, dtype=np.float64),
                kds=np.full((1, len(joint_indices)), 5000.0, dtype=np.float64),
                joint_indices=joint_indices,
            )
            print(
                f"[policy_eval] strengthened drives for {len(joint_indices)} non-gripper robot joints",
                flush=True,
            )
        except Exception as exc:
            print(f"[policy_eval] failed to strengthen robot drives: {exc}", flush=True)

    def _initialize_posture_hold_targets(self) -> None:
        gripper_names = set(self._all_gripper_joint_names())
        positions = self.robot.get_joint_positions()
        if positions is None:
            return
        self.posture_hold_targets = {
            name: float(positions[idx])
            for idx, name in enumerate(self.robot.dof_names)
            if name not in gripper_names
        }
        print(
            f"[policy_eval] posture hold enabled for {len(self.posture_hold_targets)} non-gripper robot joints",
            flush=True,
        )

    def _update_posture_hold_targets(self, joint_names: list[str], positions: np.ndarray) -> None:
        if not self.posture_hold_targets:
            return
        gripper_names = set(self._all_gripper_joint_names())
        for joint_name, position in zip(joint_names, positions):
            if joint_name in gripper_names:
                continue
            if joint_name in self.posture_hold_targets:
                self.posture_hold_targets[joint_name] = float(position)

    def _apply_posture_hold(self) -> None:
        if not self.posture_hold_targets or self.robot is None:
            return
        from isaacsim.core.utils.types import ArticulationAction

        positions = []
        indices = []
        dof_name_to_index = {name: idx for idx, name in enumerate(self.robot.dof_names)}
        for joint_name, position in self.posture_hold_targets.items():
            joint_index = dof_name_to_index.get(joint_name)
            if joint_index is None:
                continue
            positions.append(float(position))
            indices.append(int(joint_index))
        if not indices:
            return
        target_positions = np.asarray(positions, dtype=np.float64)
        target_positions = self._clip_positions(indices, target_positions)
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=target_positions,
                joint_indices=np.asarray(indices, dtype=np.int32),
            )
        )

    def _all_gripper_joint_names(self) -> list[str]:
        gripper_cfg = self.robot_cfg.get("gripper", {})
        names = []
        for values in gripper_cfg.get("finger_names", {}).values():
            names.extend(list(values))
        return names

    def _default_gripper_control(self) -> dict[str, Any]:
        return {
            "arm": "",
            "command": "",
            "mode": "idle",
            "last_sample_position": None,
            "sample_count": 0,
            "frame_count": 0,
            "hold_positions": None,
        }

    def is_gripper_holding(self, arm: str) -> bool:
        return (
            self.gripper_control.get("arm") == arm
            and self.gripper_control.get("command") == "close"
            and self.gripper_control.get("mode") == "holding"
        )

    def _update_gripper_control(self) -> None:
        mode = self.gripper_control.get("mode", "idle")
        arm = self.gripper_control.get("arm", "")
        if mode == "idle" or not arm:
            return
        if mode == "opening":
            if self._is_gripper_open(arm):
                self._apply_gripper_stop(arm)
                self.gripper_control["mode"] = "idle"
                return
            self._apply_gripper_open_velocity(arm)
            return
        if mode == "closing":
            self._apply_gripper_close_velocity(arm)
            if self._is_gripper_close_motion_settled(arm):
                self._latch_gripper_hold_positions(arm)
                self._apply_gripper_stop(arm)
                self.gripper_control["mode"] = "holding"
            return
        if mode == "holding":
            self._hold_gripper_at_current_pose(arm)

    def _apply_gripper_open_velocity(self, arm: str) -> None:
        indices = self._gripper_joint_indices(arm)
        if not indices:
            return
        current = self.robot.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return
        current = np.asarray(current, dtype=np.float64)
        opened_positions = self._gripper_open_positions(arm, len(indices))
        closed_velocity_magnitudes = np.abs(self._gripper_closed_velocities(arm, len(indices)))
        speeds = np.where(closed_velocity_magnitudes > 0.0, np.minimum(closed_velocity_magnitudes, 40.0), 40.0)
        velocities = np.sign(opened_positions - current) * speeds
        velocities[np.abs(opened_positions - current) <= 0.0025] = 0.0
        self._apply_gripper_velocity(indices, velocities)

    def _apply_gripper_close_velocity(self, arm: str) -> None:
        indices = self._gripper_joint_indices(arm)
        if not indices:
            return
        velocities = self._gripper_closed_velocities(arm, len(indices))
        self._apply_gripper_velocity(indices, velocities)

    def _apply_gripper_stop(self, arm: str) -> None:
        indices = self._gripper_joint_indices(arm)
        if indices:
            self._apply_gripper_velocity(indices, np.zeros(len(indices), dtype=np.float64))

    def _apply_gripper_velocity(self, indices: list[int], velocities: np.ndarray) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        target_velocities = [None] * len(self.robot.dof_names)
        for joint_index, velocity in zip(indices, velocities):
            target_velocities[int(joint_index)] = float(velocity)
        self.robot.apply_action(ArticulationAction(joint_velocities=target_velocities))

    def _hold_gripper_at_current_pose(self, arm: str) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        indices = self._gripper_joint_indices(arm)
        if not indices:
            return
        self._set_gripper_drive(
            arm,
            stiffness=self._config_float("gripper_hold_stiffness", 10000.0),
            max_force=self._config_float("gripper_hold_max_force", 10.0),
        )
        hold_positions = self.gripper_control.get("hold_positions")
        if hold_positions is None:
            hold_positions = self._latch_gripper_hold_positions(arm)
        if hold_positions is None or len(hold_positions) != len(indices):
            return
        target_positions = [None] * len(self.robot.dof_names)
        for joint_index, position in zip(indices, hold_positions):
            target_positions[int(joint_index)] = float(position)
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))

    def _latch_gripper_hold_positions(self, arm: str) -> np.ndarray | None:
        indices = self._gripper_joint_indices(arm)
        if not indices:
            return None
        current = self.robot.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return None
        hold_positions = np.asarray(current, dtype=np.float64).copy()
        self.gripper_control["hold_positions"] = hold_positions
        return hold_positions

    def _is_gripper_open(self, arm: str) -> bool:
        indices = self._gripper_joint_indices(arm)
        if not indices:
            return True
        current = self.robot.get_joint_positions(joint_indices=indices)
        if current is None or len(current) != len(indices):
            return False
        current = np.asarray(current, dtype=np.float64)
        target = self._gripper_open_positions(arm, len(indices))
        return bool(np.all(np.abs(current - target) <= 0.01))

    def _is_gripper_close_motion_settled(self, arm: str) -> bool:
        indices = self._gripper_joint_indices(arm)
        if not indices:
            return True
        sample_interval = max(1, int(round(0.1 * self.physics_step)))
        frame_count = int(self.gripper_control.get("frame_count", 0)) + 1
        self.gripper_control["frame_count"] = frame_count
        if frame_count % sample_interval != 0:
            return False

        current = self.robot.get_joint_positions(joint_indices=indices)
        if current is None or len(current) == 0:
            return False
        drive_position = float(current[0])
        last_position = self.gripper_control.get("last_sample_position")
        sample_count = int(self.gripper_control.get("sample_count", 0)) + 1
        self.gripper_control["sample_count"] = sample_count
        self.gripper_control["last_sample_position"] = drive_position
        if last_position is None:
            return False
        return abs(drive_position - float(last_position)) <= 0.01 or sample_count > 50

    def _set_gripper_drive(self, arm: str, stiffness: float | None = None, max_force: float | None = None) -> None:
        from pxr import UsdPhysics

        control_prim_path = self.robot_cfg.get("gripper", {}).get("gripper_controll_joint", {}).get(arm, "")
        if not control_prim_path:
            return
        prim = self.stage.GetPrimAtPath(control_prim_path)
        if not prim or not prim.IsValid():
            return
        gripper_type = self.robot_cfg.get("gripper", {}).get("gripper_type", "angular")
        drive = UsdPhysics.DriveAPI.Get(prim, gripper_type)
        if not drive:
            return
        cached = self.gripper_drive_settings.setdefault(arm, {})
        if stiffness is not None:
            stiffness = float(stiffness)
            if cached.get("stiffness") != stiffness:
                drive.GetStiffnessAttr().Set(stiffness)
                cached["stiffness"] = stiffness
        if max_force is not None:
            max_force = float(max_force)
            if cached.get("max_force") != max_force:
                drive.GetMaxForceAttr().Set(max_force)
                cached["max_force"] = max_force

    def _gripper_joint_names_for_arm(self, arm: str) -> list[str]:
        return list(self.robot_cfg.get("gripper", {}).get("finger_names", {}).get(arm, []))

    def _gripper_joint_indices(self, arm: str) -> list[int]:
        names = self._gripper_joint_names_for_arm(arm)
        if not names:
            return []
        return self._joint_indices(names)

    def _gripper_open_positions(self, arm: str, count: int) -> np.ndarray:
        positions = self.robot_cfg.get("gripper", {}).get("opened_positions", {}).get(arm, [])
        return _fit_vector(positions, count, default=0.0)

    def _gripper_closed_velocities(self, arm: str, count: int) -> np.ndarray:
        velocities = self.robot_cfg.get("gripper", {}).get("closed_velocities", {}).get(arm, [])
        return _fit_vector(velocities, count, default=80.0)


def _resolve_asset_path(assets_root: Path, path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    candidate = assets_root / value
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        for suffix in ("Aligned.usd", "Aligned.usda"):
            nested = candidate / suffix
            if nested.exists():
                return nested
    return candidate


def _resolve_object_asset_path(assets_root: Path, obj: dict[str, Any]) -> Path:
    data_info_dir = obj.get("data_info_dir")
    if data_info_dir:
        data_info_path = Path(data_info_dir)
        if not data_info_path.is_absolute():
            data_info_path = assets_root / data_info_path
        for filename in ("Aligned.usd", "Aligned.usda"):
            candidate = data_info_path / filename
            if candidate.exists():
                return candidate

    model_path = obj.get("model_path", "")
    candidate = _resolve_asset_path(assets_root, model_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Object asset not found for {obj.get('object_id')}: {candidate}")


def _resolve_target_object_id(task_instance: dict[str, Any]) -> str:
    for stage in task_instance.get("stages", []):
        if stage.get("action") != "pick":
            continue
        passive = stage.get("passive", {})
        object_id = passive.get("object_id", "")
        if object_id and object_id != "gripper":
            return object_id
    for obj in task_instance.get("objects", []):
        if obj.get("is_key") and "target" in obj.get("object_id", ""):
            return obj.get("object_id", "")
    return ""


def _resolve_place_target_object_id(task_instance: dict[str, Any]) -> str:
    for stage in task_instance.get("stages", []):
        if stage.get("action") != "place":
            continue
        passive = stage.get("passive", {})
        object_id = passive.get("object_id", "")
        if object_id and object_id != "gripper":
            return object_id
    for rule in task_instance.get("task_metric", {}).get("filter_rules", []):
        params = rule.get("params", {})
        target = params.get("target") or params.get("target_id", "")
        if target and target != "gripper":
            return target
    return ""


def _resolve_required_task_object_ids(task_instance: dict[str, Any]) -> set[str]:
    object_ids = set()
    for stage in task_instance.get("stages", []):
        if stage.get("action") not in ("pick", "place"):
            continue
        for role in ("active", "passive"):
            object_id = stage.get(role, {}).get("object_id", "")
            if object_id and object_id != "gripper":
                object_ids.add(object_id)
    for rule in task_instance.get("task_metric", {}).get("filter_rules", []):
        params = rule.get("params", {})
        for object_id in params.get("objects", []):
            if object_id and object_id != "gripper":
                object_ids.add(object_id)
        for key in ("object_id", "target", "target_id"):
            object_id = params.get(key, "")
            if object_id and object_id != "gripper":
                object_ids.add(object_id)
    return object_ids


def _resolve_object_scale(obj: dict[str, Any]) -> list[float]:
    if "scale" not in obj:
        size = np.asarray(obj.get("size", []), dtype=np.float64)
        scale = 0.001 if size.size > 0 and float(np.max(size)) > 10.0 else 1.0
    else:
        scale = obj["scale"]
    return _normalize_scale(scale)


def _normalize_scale(scale: Any) -> list[float]:
    if scale is None:
        return [1.0, 1.0, 1.0]
    if isinstance(scale, (int, float)):
        return [float(scale), float(scale), float(scale)]
    values = list(scale)
    if len(values) == 1:
        return [float(values[0])] * 3
    if len(values) != 3:
        raise ValueError(f"Object scale must be a scalar or length-3 sequence, got {scale}")
    return [float(value) for value in values]


def _optional_vector(value: Any, expected_len: int) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (expected_len,):
        raise ValueError(f"Expected a length-{expected_len} vector, got {value}")
    return vector


def _fit_vector(value: Any, count: int, default: float) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.float64)
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return np.full((count,), float(default), dtype=np.float64)
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return np.full((count,), float(default), dtype=np.float64)
    if vector.size == 1 and count > 1:
        return np.full((count,), float(vector[0]), dtype=np.float64)
    if vector.size < count:
        tail = np.full((count - vector.size,), float(vector[-1]), dtype=np.float64)
        return np.concatenate([vector, tail])
    return vector[:count].astype(np.float64)


def _look_at_quat_world_axes(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError(f"Observer camera position and target are identical: {position.tolist()}")
    forward = forward / forward_norm

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    z_axis = up_hint - forward * float(np.dot(up_hint, forward))
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, forward)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((forward, y_axis, z_axis))
    return matrix_to_quat_wxyz(rotation)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)
