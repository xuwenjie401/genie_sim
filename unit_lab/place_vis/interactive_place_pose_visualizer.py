#!/usr/bin/env python3
"""Interactive Isaac Sim visualizer for Place pose generation.

Features:
- Load active object (to be placed) and passive object (container)
- Visualize place pose generation using REAL place.py / common.py logic (get_aligned_pose)
- Interactively adjust passive place annotation (xyz, direction, angle_sample_num)
- See real-time updates of generated candidate poses
- Switch between storage boxes to compare IK-candidate distributions

Usage:
    python unit_lab/place_vis/interactive_place_pose_visualizer.py \
        --active_object benchmark_beverage_bottle_001 \
        --passive_object benchmark_storage_box_011
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

from omni.isaac.kit import SimulationApp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_ASSET_ROOT = Path("/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets")
DEFAULT_OBJECTS_ROOT = DEFAULT_ASSET_ROOT / "objects" / "benchmark"
DEFAULT_INTERACTION_ROOT = DEFAULT_ASSET_ROOT / "interaction"

DEFAULT_ACTIVE_OBJECT = "benchmark_beverage_bottle_001"
DEFAULT_PASSIVE_OBJECT = "benchmark_storage_box_001"

SCENE_ROOT = "/World/PlacePoseVis"
PASSIVE_OBJ_PATH = f"{SCENE_ROOT}/PassiveObject"
ACTIVE_OBJ_PATH = f"{SCENE_ROOT}/ActiveObject"
POSES_ROOT = f"{SCENE_ROOT}/CandidatePoses"
LIGHT_PATH = f"{SCENE_ROOT}/KeyLight"
UI_TITLE = "Interactive Place Pose Visualizer"

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--objects_root", type=Path, default=DEFAULT_OBJECTS_ROOT)
    p.add_argument("--interaction_root", type=Path, default=DEFAULT_INTERACTION_ROOT)
    p.add_argument("--active_object", type=str, default=DEFAULT_ACTIVE_OBJECT)
    p.add_argument("--passive_object", type=str, default=DEFAULT_PASSIVE_OBJECT)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--max_display_poses", type=int, default=80,
                   help="Max candidate poses to display")
    p.add_argument("--N_align", type=int, default=0,
                   help="Override angle sample count (0 = use lcm from annotations)")
    return p.parse_known_args()[0]


ARGS = parse_args()
simulation_app = SimulationApp({"width": ARGS.width, "height": ARGS.height, "headless": False})

# ---------------------------------------------------------------------------
# Imports that need Isaac Sim to be up first
# ---------------------------------------------------------------------------
import numpy as np
import omni.ui as ui
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.prims import create_prim
from pxr import Gf, UsdGeom, UsdLux

try:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
except Exception:
    ScrollingWindow = None

# Real pipeline objects
from source.data_collection.client.layout.object import OmniObject
from source.data_collection.client.planner.common import get_aligned_pose


# ---------------------------------------------------------------------------
# USD helpers (copied from interactive_interaction_pose_editor.py)
# ---------------------------------------------------------------------------
def remove_prim_if_exists(stage, prim_path: str) -> None:
    p = stage.GetPrimAtPath(prim_path)
    if p and p.IsValid():
        stage.RemovePrim(prim_path)


def ensure_prim(stage, prim_path: str, prim_type: str = "Xform"):
    p = stage.GetPrimAtPath(prim_path)
    if p and p.IsValid():
        return p
    return create_prim(prim_path, prim_type=prim_type)


def set_display_color(prim, rgb: np.ndarray, opacity: float = 1.0) -> None:
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    g = UsdGeom.Gprim(prim)
    g.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    g.CreateDisplayOpacityAttr().Set([float(max(0.0, min(opacity, 1.0)))])


def np_to_gf_matrix4d(m: np.ndarray) -> Gf.Matrix4d:
    t = np.asarray(m, dtype=np.float64).T
    return Gf.Matrix4d(
        t[0,0], t[0,1], t[0,2], t[0,3],
        t[1,0], t[1,1], t[1,2], t[1,3],
        t[2,0], t[2,1], t[2,2], t[2,3],
        t[3,0], t[3,1], t[3,2], t[3,3],
    )


def set_local_matrix(prim, m: np.ndarray) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(np_to_gf_matrix4d(m))


def set_local_translate_scale(prim, t_xyz, s_xyz) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t_op = xf.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, opSuffix="t")
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))
    s_op = xf.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, opSuffix="s")
    s_op.Set(Gf.Vec3d(float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])))


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Load object info from disk
# ---------------------------------------------------------------------------
def load_object_params(object_dir: Path) -> dict:
    f = object_dir / "object_parameters.json"
    if not f.exists():
        return {"size": [0.1, 0.1, 0.1], "scale": 1.0}
    with f.open() as fh:
        d = json.load(fh)
    return d


def find_usd(object_dir: Path) -> Path | None:
    for name in ("Aligned.usda", "Aligned.usd"):
        c = object_dir / name
        if c.exists():
            return c
    return None


def load_interaction(interaction_dir: Path) -> dict:
    f = interaction_dir / "interaction.json"
    if not f.exists():
        return {}
    with f.open() as fh:
        return json.load(fh)


def find_object_dir(objects_root: Path, object_id: str) -> Path | None:
    for cat_dir in objects_root.iterdir():
        if not cat_dir.is_dir():
            continue
        candidate = cat_dir / object_id
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Build a simple axis-frame marker (3 colored sticks)
# ---------------------------------------------------------------------------
AXIS_LEN = 0.04
AXIS_THICK = 0.002


def build_frame_marker(stage, root_path: str, scale: float = 1.0) -> None:
    """Draw XYZ axes at root_path (X=red, Y=green, Z=blue)."""
    axes = [
        ("x", (AXIS_LEN * scale / 2, 0, 0), (AXIS_LEN * scale, AXIS_THICK, AXIS_THICK), np.array([1, 0, 0], dtype=np.float32)),
        ("y", (0, AXIS_LEN * scale / 2, 0), (AXIS_THICK, AXIS_LEN * scale, AXIS_THICK), np.array([0, 1, 0], dtype=np.float32)),
        ("z", (0, 0, AXIS_LEN * scale / 2), (AXIS_THICK, AXIS_THICK, AXIS_LEN * scale), np.array([0, 0, 1], dtype=np.float32)),
    ]
    for name, t, s, color in axes:
        p = create_prim(f"{root_path}/{name}", prim_type="Cube")
        UsdGeom.Cube(p).CreateSizeAttr().Set(1.0)
        set_local_translate_scale(p, t, s)
        set_display_color(p, color)


def build_sphere_marker(stage, path: str, radius: float, color: np.ndarray, opacity: float = 0.85) -> None:
    p = create_prim(path, prim_type="Sphere")
    UsdGeom.Sphere(p).CreateRadiusAttr().Set(float(radius))
    set_display_color(p, color, opacity)


def build_arrow_marker(stage, path: str, direction: np.ndarray, length: float, color: np.ndarray) -> None:
    """Draw a thin cylinder along `direction` of given length."""
    d = normalize(direction)
    # Build rotation that maps Z → d
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        R = np.eye(4)
        if float(np.dot(z, d)) < 0:
            R[2, 2] = -1
    else:
        axis = axis / axis_norm
        angle = float(np.arccos(np.clip(float(np.dot(z, d)), -1, 1)))
        c, s = np.cos(angle), np.sin(angle)
        x, y, zz = axis
        R3 = np.array([
            [c + x*x*(1-c),   x*y*(1-c) - zz*s, x*zz*(1-c) + y*s],
            [y*x*(1-c) + zz*s, c + y*y*(1-c),   y*zz*(1-c) - x*s],
            [zz*x*(1-c) - y*s, zz*y*(1-c) + x*s, c + zz*zz*(1-c)],
        ])
        R = np.eye(4)
        R[:3, :3] = R3
    R[:3, 3] = d * length / 2
    prim = create_prim(path, prim_type="Cylinder")
    UsdGeom.Cylinder(prim).CreateHeightAttr().Set(float(length))
    UsdGeom.Cylinder(prim).CreateRadiusAttr().Set(0.003)
    set_local_matrix(prim, R)
    set_display_color(prim, color, 0.9)


# ---------------------------------------------------------------------------
# Visualize a single candidate pose as a small coordinate frame + sphere
# ---------------------------------------------------------------------------
POSE_COLORS = [
    np.array([0.2, 0.7, 1.0], dtype=np.float32),   # light blue  – most candidates
    np.array([1.0, 0.7, 0.1], dtype=np.float32),   # orange      – top-ranked
    np.array([0.2, 1.0, 0.4], dtype=np.float32),   # green       – extra
]


def build_candidate_pose_vis(stage, root_path: str, pose_4x4: np.ndarray, idx: int, total: int) -> None:
    """Visualize one candidate object pose as a small axis frame + dot."""
    create_prim(root_path, prim_type="Xform")
    # Position marker
    phase = idx / max(total - 1, 1)
    color = np.clip(
        POSE_COLORS[0] * (1 - phase) + POSE_COLORS[1] * phase,
        0, 1
    ).astype(np.float32)
    build_sphere_marker(stage, f"{root_path}/dot", radius=0.008, color=color, opacity=0.8)
    # Axes
    frame_prim = create_prim(f"{root_path}/frame", prim_type="Xform")
    build_frame_marker(stage, f"{root_path}/frame")
    # Place the whole group at pose
    set_local_matrix(stage.GetPrimAtPath(root_path), pose_4x4)


# ---------------------------------------------------------------------------
# Core visualizer class
# ---------------------------------------------------------------------------
class PlacePoseVisualizer:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()

        # Current object IDs
        self.active_id = ARGS.active_object
        self.passive_id = ARGS.passive_object

        # Loaded data
        self._active_omni: OmniObject | None = None
        self._passive_omni: OmniObject | None = None
        self._active_usd: Path | None = None
        self._passive_usd: Path | None = None

        # Passive place elements (list of {xyz, direction, ...})
        self._passive_primitives: dict[str, list[dict]] = {}   # primitive_name -> items
        self._active_elements: list[dict] = []                  # active place elements
        self._current_primitive = "default"
        self._selected_item_idx = 0  # which passive element item is selected

        # UI float models  (xyz + direction for selected passive element)
        self._float_models: dict[str, ui.SimpleFloatModel] = {}
        self._n_align_model: ui.SimpleStringModel | None = None
        self._suspend_callbacks = False

        # Status
        self.status = "Initializing..."
        self._scene_dirty = True
        self._ui_dirty = True
        self._window = None

        # Passive object world pose (placed slightly away from origin)
        self._passive_world_pose = np.eye(4, dtype=np.float64)
        self._passive_world_pose[1, 3] = 0.35   # move 35 cm in Y

        # Active object world pose (on table, to the side)
        self._active_world_pose = np.eye(4, dtype=np.float64)
        self._active_world_pose[0, 3] = -0.30
        self._active_world_pose[1, 3] = 0.35

        self._warm_up()
        self._load_objects()
        self._ensure_scene()
        self._ensure_light()
        self._create_float_models()
        self._build_ui()
        self._apply_scene(force=True)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _warm_up(self) -> None:
        for _ in range(20):
            simulation_app.update()

    def _ensure_scene(self) -> None:
        ensure_prim(self.stage, "/World", "Xform")
        ensure_prim(self.stage, SCENE_ROOT, "Xform")
        ensure_prim(self.stage, PASSIVE_OBJ_PATH, "Xform")
        ensure_prim(self.stage, ACTIVE_OBJ_PATH, "Xform")
        ensure_prim(self.stage, POSES_ROOT, "Xform")

    def _ensure_light(self) -> None:
        if not self.stage.GetPrimAtPath(LIGHT_PATH):
            light = UsdLux.SphereLight.Define(self.stage, LIGHT_PATH)
            light.CreateIntensityAttr(70000.0)
            light.CreateRadiusAttr(0.3)
            xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(LIGHT_PATH))
            op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(1.0, 1.4, 1.8))

    def _load_object_data(self, object_id: str) -> tuple[Path | None, dict, dict]:
        """Returns (usd_path, params, interaction_data)."""
        obj_dir = find_object_dir(ARGS.objects_root, object_id)
        if obj_dir is None:
            print(f"[Vis] Object dir not found: {object_id}")
            return None, {}, {}
        usd = find_usd(obj_dir)
        params = load_object_params(obj_dir)
        interaction_dir = ARGS.interaction_root / object_id
        interaction = load_interaction(interaction_dir)
        return usd, params, interaction

    def _make_omni_object(self, object_id: str, params: dict, interaction: dict,
                          world_pose: np.ndarray) -> OmniObject:
        size = params.get("size", [0.1, 0.1, 0.1])
        scale = params.get("scale", 1.0)
        size_scaled = [s * scale for s in size]
        obj = OmniObject(name=object_id, size=size_scaled, pose=world_pose.copy())
        obj.elements = interaction.get("interaction", {})
        return obj

    def _load_objects(self) -> None:
        # Active
        usd_a, params_a, inter_a = self._load_object_data(self.active_id)
        self._active_usd = usd_a
        self._active_omni = self._make_omni_object(
            self.active_id, params_a, inter_a, self._active_world_pose)

        # Passive
        usd_p, params_p, inter_p = self._load_object_data(self.passive_id)
        self._passive_usd = usd_p
        self._passive_omni = self._make_omni_object(
            self.passive_id, params_p, inter_p, self._passive_world_pose)

        # Parse place elements
        self._parse_place_elements()
        self.status = f"Loaded: active={self.active_id}  passive={self.passive_id}"
        print(f"[Vis] {self.status}")

    def _parse_place_elements(self) -> None:
        """Extract place annotation dicts from the interaction data."""
        # Passive place primitives
        passive_inter = self._passive_omni.elements
        passive_place = (passive_inter.get("passive") or {}).get("place") or {}
        self._passive_primitives = {}
        for prim_name, items in passive_place.items():
            if isinstance(items, list):
                valid = []
                for it in items:
                    if isinstance(it, dict) and "xyz" in it and "direction" in it:
                        valid.append({
                            "xyz": np.array(it["xyz"], dtype=np.float64),
                            "direction": np.array(it["direction"], dtype=np.float64),
                            "angle_sample_num": int(it.get("angle_sample_num", 72)),
                        })
                if valid:
                    self._passive_primitives[prim_name] = valid

        # Active place elements (all primitives merged)
        active_inter = self._active_omni.elements
        active_place = (active_inter.get("active") or {}).get("place") or {}
        self._active_elements = []
        for prim_name, items in active_place.items():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and "xyz" in it and "direction" in it:
                        self._active_elements.append({
                            "xyz": np.array(it["xyz"], dtype=np.float64),
                            "direction": np.array(it["direction"], dtype=np.float64),
                            "angle_sample_num": int(it.get("angle_sample_num", 72)),
                        })

        # Default selection
        if self._passive_primitives:
            self._current_primitive = next(iter(self._passive_primitives))
        self._selected_item_idx = 0

        print(f"[Vis] Passive primitives: {list(self._passive_primitives.keys())}")
        print(f"[Vis] Active elements: {len(self._active_elements)}")

    # ------------------------------------------------------------------
    # Float models (xyz + direction of selected passive element)
    # ------------------------------------------------------------------
    def _create_float_models(self) -> None:
        for key in ("xyz.x", "xyz.y", "xyz.z", "dir.x", "dir.y", "dir.z"):
            m = ui.SimpleFloatModel(0.0)
            m.add_end_edit_fn(lambda _m, k=key: self._on_float_changed(k, _m))
            self._float_models[key] = m
        self._n_align_model = ui.SimpleStringModel("72")
        self._sync_models_from_element()

    def _current_passive_item(self) -> dict | None:
        items = self._passive_primitives.get(self._current_primitive, [])
        if not items:
            return None
        self._selected_item_idx = max(0, min(self._selected_item_idx, len(items) - 1))
        return items[self._selected_item_idx]

    def _sync_models_from_element(self) -> None:
        self._suspend_callbacks = True
        item = self._current_passive_item()
        if item is None:
            self._suspend_callbacks = False
            return
        xyz = item["xyz"]
        d = item["direction"]
        self._float_models["xyz.x"].set_value(float(xyz[0]))
        self._float_models["xyz.y"].set_value(float(xyz[1]))
        self._float_models["xyz.z"].set_value(float(xyz[2]))
        self._float_models["dir.x"].set_value(float(d[0]))
        self._float_models["dir.y"].set_value(float(d[1]))
        self._float_models["dir.z"].set_value(float(d[2]))
        self._n_align_model.set_value(str(item["angle_sample_num"]))
        self._suspend_callbacks = False
        self._ui_dirty = True

    def _on_float_changed(self, key: str, model: ui.SimpleFloatModel) -> None:
        if self._suspend_callbacks:
            return
        item = self._current_passive_item()
        if item is None:
            return
        val = float(model.get_value_as_float())
        if key == "xyz.x":   item["xyz"][0] = val
        elif key == "xyz.y": item["xyz"][1] = val
        elif key == "xyz.z": item["xyz"][2] = val
        elif key == "dir.x": item["direction"][0] = val
        elif key == "dir.y": item["direction"][1] = val
        elif key == "dir.z": item["direction"][2] = val
        self._scene_dirty = True
        self._ui_dirty = True

    def _apply_n_align_edit(self) -> None:
        item = self._current_passive_item()
        if item is None:
            return
        try:
            v = int(self._n_align_model.get_value_as_string())
            item["angle_sample_num"] = max(1, v)
        except ValueError:
            pass
        self._scene_dirty = True
        self._ui_dirty = True

    # ------------------------------------------------------------------
    # Core pose generation (real pipeline logic)
    # ------------------------------------------------------------------
    def _generate_candidate_poses(self) -> np.ndarray:
        """
        Reproduce the place.py select_pose() pose-generation loop using the
        real get_aligned_pose() from client.planner.common.

        Returns array of shape (N, 4, 4) — active object poses in WORLD frame.
        """
        if self._active_omni is None or self._passive_omni is None:
            return np.zeros((0, 4, 4))
        if not self._active_elements:
            return np.zeros((0, 4, 4))

        passive_items = self._passive_primitives.get(self._current_primitive, [])
        if not passive_items:
            return np.zeros((0, 4, 4))

        all_poses = []
        for active_el in self._active_elements:
            # Update active obj aligned info (mirrors place.py:111)
            self._active_omni.update_aligned_info(active_el)
            self._active_omni.set_pose(self._active_world_pose, self._active_omni.obj_length)

            for passive_el in passive_items:
                # Update passive obj aligned info (mirrors place.py:114)
                self._passive_omni.update_aligned_info(passive_el)
                self._passive_omni.set_pose(self._passive_world_pose, self._passive_omni.obj_length)

                # N_align: lcm of sample counts (mirrors place.py:116)
                if ARGS.N_align > 0:
                    N_align = ARGS.N_align
                else:
                    N_align = int(np.lcm(
                        self._passive_omni.angle_sample_num,
                        self._active_omni.angle_sample_num,
                    ))

                # Call REAL get_aligned_pose (mirrors place.py:123)
                try:
                    poses = get_aligned_pose(
                        self._active_omni,
                        self._passive_omni,
                        N=N_align,
                    )  # shape (N_align, 4, 4) in world frame
                    all_poses.append(poses)
                except Exception as e:
                    print(f"[Vis] get_aligned_pose failed: {e}")

        if not all_poses:
            return np.zeros((0, 4, 4))
        return np.concatenate(all_poses, axis=0)

    # ------------------------------------------------------------------
    # Scene update
    # ------------------------------------------------------------------
    def _load_usd_asset(self, parent_path: str, usd_path: Path | None,
                        world_pose: np.ndarray) -> None:
        """Reference a USD asset under parent_path and set its world transform."""
        # Clear previous assets
        parent = self.stage.GetPrimAtPath(parent_path)
        if parent and parent.IsValid():
            for child in list(parent.GetChildren()):
                self.stage.RemovePrim(str(child.GetPath()))
        for _ in range(2):
            simulation_app.update()

        if usd_path is None:
            return

        asset_prim = create_prim(f"{parent_path}/Asset", prim_type="Xform")
        asset_prim.GetReferences().AddReference(str(usd_path))
        set_local_matrix(asset_prim, world_pose)
        for _ in range(4):
            simulation_app.update()

    def _rebuild_candidate_poses(self) -> None:
        remove_prim_if_exists(self.stage, POSES_ROOT)
        create_prim(POSES_ROOT, prim_type="Xform")

        poses = self._generate_candidate_poses()
        n = min(poses.shape[0], ARGS.max_display_poses)
        print(f"[Vis] Generated {poses.shape[0]} candidate poses, displaying {n}")
        self.status = (
            f"Candidate poses: {poses.shape[0]}  (showing {n}) | "
            f"passive={self.passive_id}  primitive={self._current_primitive}  "
            f"item={self._selected_item_idx + 1}/{len(self._passive_primitives.get(self._current_primitive, []))}"
        )

        for i in range(n):
            pose_path = f"{POSES_ROOT}/pose_{i:04d}"
            build_candidate_pose_vis(self.stage, pose_path, poses[i], i, n)

        for _ in range(2):
            simulation_app.update()

    def _rebuild_annotation_markers(self) -> None:
        """Draw the current passive place element xyz+direction as a marker."""
        marker_root = f"{SCENE_ROOT}/AnnotationMarker"
        remove_prim_if_exists(self.stage, marker_root)
        item = self._current_passive_item()
        if item is None:
            return
        create_prim(marker_root, prim_type="Xform")

        # xyz in passive object local frame → world frame
        xyz_local = np.array(item["xyz"], dtype=np.float64)
        direction_local = normalize(np.array(item["direction"], dtype=np.float64))

        # Transform to world
        xyz_world = (self._passive_world_pose[:3, :3] @ xyz_local) + self._passive_world_pose[:3, 3]
        dir_world = self._passive_world_pose[:3, :3] @ direction_local

        # Dot at xyz (yellow)
        dot = create_prim(f"{marker_root}/dot", prim_type="Sphere")
        UsdGeom.Sphere(dot).CreateRadiusAttr().Set(0.012)
        xf = UsdGeom.Xformable(dot)
        xf.ClearXformOpOrder()
        op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        op.Set(Gf.Vec3d(float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])))
        set_display_color(dot, np.array([1.0, 1.0, 0.0], dtype=np.float32), 1.0)

        # Arrow along direction (magenta)
        build_arrow_marker(self.stage, f"{marker_root}/dir_arrow",
                           direction=dir_world, length=0.08,
                           color=np.array([1.0, 0.2, 0.8], dtype=np.float32))
        # Translate arrow to start at xyz_world
        arrow_prim = self.stage.GetPrimAtPath(f"{marker_root}/dir_arrow")
        if arrow_prim and arrow_prim.IsValid():
            # Arrow is built centered at origin along dir; shift it
            current_mat_op = UsdGeom.Xformable(arrow_prim).GetOrderedXformOps()
            # Rebuild with correct translation
            remove_prim_if_exists(self.stage, f"{marker_root}/dir_arrow")
            build_arrow_marker(self.stage, f"{marker_root}/dir_arrow",
                               direction=dir_world, length=0.08,
                               color=np.array([1.0, 0.2, 0.8], dtype=np.float32))
            # Apply translation offset
            a = self.stage.GetPrimAtPath(f"{marker_root}/dir_arrow")
            if a and a.IsValid():
                cur = UsdGeom.Xformable(a)
                ops = cur.GetOrderedXformOps()
                if ops:
                    m = np_to_gf_matrix4d(np.eye(4))
                    # get existing matrix
                    import pxr.Usd as Usd
                    world_m = cur.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                # Simply add translate
                op2 = cur.AddXformOp(UsdGeom.XformOp.TypeTranslate,
                                     UsdGeom.XformOp.PrecisionDouble, opSuffix="shift")
                op2.Set(Gf.Vec3d(float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2])))

    def _apply_scene(self, force: bool = False) -> None:
        if force:
            self._load_usd_asset(PASSIVE_OBJ_PATH, self._passive_usd, self._passive_world_pose)
            self._load_usd_asset(ACTIVE_OBJ_PATH, self._active_usd, self._active_world_pose)
            # Set camera
            set_camera_view(
                eye=[0.8, -0.5, 0.8],
                target=[0.0, 0.35, 0.0],
                camera_prim_path="/OmniverseKit_Persp",
            )
        self._rebuild_candidate_poses()
        self._rebuild_annotation_markers()
        self._scene_dirty = False
        self._ui_dirty = True

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        kwargs = dict(title=UI_TITLE, width=430, height=820, visible=True,
                      dockPreference=ui.DockPreference.LEFT_BOTTOM)
        if ScrollingWindow is not None:
            self._window = ScrollingWindow(**kwargs)
        else:
            self._window = ui.Window(**kwargs)
        self._window.visible = True
        self._rebuild_ui()

    def _rebuild_ui(self) -> None:
        if self._window is None:
            return
        frame = self._window.frame
        if hasattr(frame, "set_build_fn"):
            frame.set_build_fn(self._build_ui_contents)
        if hasattr(frame, "rebuild"):
            frame.rebuild()
        else:
            with frame:
                self._build_ui_contents()
        self._ui_dirty = False

    def _build_ui_contents(self) -> None:
        passive_items = self._passive_primitives.get(self._current_primitive, [])
        n_items = len(passive_items)
        n_active = len(self._active_elements)
        primitives = list(self._passive_primitives.keys())

        with ui.VStack(spacing=6, height=0):
            ui.Label(UI_TITLE, height=22)
            ui.Label(self.status, word_wrap=True, height=52)
            ui.Separator(height=4)

            # ---- Object selection ----
            ui.Label("Objects", height=18)
            with ui.HStack(height=26, spacing=4):
                ui.Label("Active:", width=55)
                ui.Label(self.active_id, word_wrap=True)
            with ui.HStack(height=26, spacing=4):
                ui.Label("Passive:", width=55)
                ui.Label(self.passive_id, word_wrap=True)

            ui.Separator(height=4)

            # ---- Info ----
            ui.Label(
                f"Active place elements: {n_active} | "
                f"Passive primitives: {primitives}",
                word_wrap=True, height=36)

            ui.Separator(height=4)

            # ---- Primitive selector ----
            ui.Label("Passive Primitive", height=18)
            with ui.HStack(height=28, spacing=6):
                ui.Button("<", width=28,
                          clicked_fn=lambda: self._cycle_primitive(-1))
                ui.Label(self._current_primitive, width=200)
                ui.Button(">", width=28,
                          clicked_fn=lambda: self._cycle_primitive(1))

            # ---- Passive element item selector ----
            ui.Label(f"Passive Element Item  ({self._selected_item_idx + 1}/{n_items})", height=18)
            with ui.HStack(height=28, spacing=6):
                ui.Button("<", width=28,
                          clicked_fn=lambda: self._cycle_item(-1))
                ui.Label(f"item {self._selected_item_idx + 1}", width=200)
                ui.Button(">", width=28,
                          clicked_fn=lambda: self._cycle_item(1))

            ui.Separator(height=4)

            # ---- Editable fields ----
            ui.Label("Edit Passive Element  (xyz = place point in obj-local frame)", height=18)
            self._build_vec_row("xyz", ["xyz.x", "xyz.y", "xyz.z"])
            ui.Label("direction = approach direction (passive object local frame)", height=18)
            self._build_vec_row("dir", ["dir.x", "dir.y", "dir.z"])

            with ui.HStack(height=26, spacing=4):
                ui.Label("angle_sample_num:", width=140)
                if hasattr(ui, "StringField"):
                    ui.StringField(model=self._n_align_model, width=80)
                ui.Button("Apply", width=70, clicked_fn=self._apply_n_align_edit)

            ui.Separator(height=4)

            with ui.HStack(height=30, spacing=8):
                ui.Button("Refresh Poses", width=130,
                          clicked_fn=self._force_refresh)
                ui.Button("Add Item", width=100,
                          clicked_fn=self._add_item)
                ui.Button("Del Item", width=100,
                          clicked_fn=self._del_item)

            ui.Separator(height=4)
            ui.Label(
                "Yellow dot = place point (xyz) in world frame.\n"
                "Magenta arrow = direction.\n"
                "Blue→orange frames = candidate active-object poses.\n"
                "All poses are generated by get_aligned_pose() — identical to place.py.",
                word_wrap=True, height=72)

            ui.Separator(height=4)
            ui.Label("Hint: edit a value then press Tab/Enter. Refresh updates scene.", word_wrap=True, height=36)

    def _build_vec_row(self, label: str, keys: list[str]) -> None:
        axes = ("x", "y", "z")
        with ui.HStack(height=26, spacing=4):
            ui.Label(label, width=30)
            for ax, key in zip(axes, keys):
                ui.Label(ax, width=12)
                ui.FloatField(model=self._float_models[key], width=82)

    # ---- Cycle helpers ----
    def _cycle_primitive(self, delta: int) -> None:
        primitives = list(self._passive_primitives.keys())
        if not primitives:
            return
        idx = primitives.index(self._current_primitive) if self._current_primitive in primitives else 0
        self._current_primitive = primitives[(idx + delta) % len(primitives)]
        self._selected_item_idx = 0
        self._sync_models_from_element()
        self._scene_dirty = True
        self._ui_dirty = True

    def _cycle_item(self, delta: int) -> None:
        items = self._passive_primitives.get(self._current_primitive, [])
        if not items:
            return
        self._selected_item_idx = (self._selected_item_idx + delta) % len(items)
        self._sync_models_from_element()
        self._scene_dirty = True
        self._ui_dirty = True

    def _force_refresh(self) -> None:
        self._scene_dirty = True
        self._ui_dirty = True

    def _add_item(self) -> None:
        items = self._passive_primitives.setdefault(self._current_primitive, [])
        items.append({
            "xyz": np.array([0.0, 0.02, 0.0]),
            "direction": np.array([0.0, -1.0, 0.0]),
            "angle_sample_num": 72,
        })
        self._selected_item_idx = len(items) - 1
        self._sync_models_from_element()
        self._scene_dirty = True
        self._ui_dirty = True

    def _del_item(self) -> None:
        items = self._passive_primitives.get(self._current_primitive, [])
        if not items:
            return
        items.pop(self._selected_item_idx)
        self._selected_item_idx = max(0, self._selected_item_idx - 1)
        self._sync_models_from_element()
        self._scene_dirty = True
        self._ui_dirty = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("=" * 70)
        print(UI_TITLE)
        print(f"  active  : {self.active_id}")
        print(f"  passive : {self.passive_id}")
        print(f"  passive place primitives: {list(self._passive_primitives.keys())}")
        print(f"  active  place elements  : {len(self._active_elements)}")
        print("=" * 70)

        while simulation_app.is_running():
            if self._scene_dirty:
                self._apply_scene()
            if self._ui_dirty:
                self._rebuild_ui()
            simulation_app.update()

        simulation_app.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    vis = PlacePoseVisualizer()
    vis.run()


if __name__ == "__main__":
    main()
