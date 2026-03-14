"""
Isaac Sim grasp axis visualizer with persistent geometry.

This viewer keeps the USD asset visible and updates grasp overlays in place,
so changing approach / width / top is immediate enough for alignment checks.

Controls (viewport must have focus):
    A / D or Left / Right : cycle approach axis
    W / S or Up / Down    : cycle width axis
    Q / E                 : cycle top axis
    R / F / T             : flip approach / width / top sign
    X                     : reset to (+x, +y, +z)
    H                     : print help
    Esc                    quit
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass

import isaacsim  # noqa: F401

from omni.isaac.kit import SimulationApp


parser = argparse.ArgumentParser(description="Interactive Isaac Sim grasp axis visualizer")
parser.add_argument(
    "--pkl",
    type=str,
    default=(
        "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/"
        "GenieSimAssets/interaction/benchmark_beverage_bottle_008/"
        "grasp_pose/grasp_pose.pkl"
    ),
)
parser.add_argument(
    "--usd",
    type=str,
    default=(
        "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/"
        "GenieSimAssets/objects/benchmark/beverage_bottle/"
        "benchmark_beverage_bottle_008/Aligned.usda"
    ),
)
parser.add_argument("--object_path", type=str, default="/World/Aligned")
parser.add_argument("--max_grasps", type=int, default=16)
parser.add_argument("--finger_len", type=float, default=0.04)
parser.add_argument("--handle_len", type=float, default=0.08)
parser.add_argument("--thickness", type=float, default=0.003)
parser.add_argument("--axis_len", type=float, default=0.05)
parser.add_argument("--axis_thickness", type=float, default=0.0016)
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"width": 1920, "height": 1080, "headless": False})

import carb
import carb.input
import omni.appwindow
import omni.usd
import numpy as np
from omni.isaac.core.utils.prims import create_prim
from pxr import Gf, UsdGeom


AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
AXIS_VECS = {
    "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "+z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}
ANSI = {"x": "\033[31m", "y": "\033[32m", "z": "\033[34m"}
ANSI_RESET = "\033[0m"


def axis_base(axis_name: str) -> str:
    return axis_name[-1]


def set_display_color(prim, rgb: np.ndarray) -> None:
    color = np.asarray(rgb, dtype=np.float32).reshape(3)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )


def remove_prim_if_exists(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def np_to_gf_matrix4d(T: np.ndarray) -> Gf.Matrix4d:
    M = np.asarray(T, dtype=np.float64).T
    return Gf.Matrix4d(
        M[0, 0], M[0, 1], M[0, 2], M[0, 3],
        M[1, 0], M[1, 1], M[1, 2], M[1, 3],
        M[2, 0], M[2, 1], M[2, 2], M[2, 3],
        M[3, 0], M[3, 1], M[3, 2], M[3, 3],
    )


def set_local_matrix(prim, T_local_4x4: np.ndarray) -> None:
    xformable = UsdGeom.Xformable(prim)
    transform_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = op
            break
    if transform_op is None:
        transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(np_to_gf_matrix4d(T_local_4x4))


def set_local_translate_scale(prim, t_xyz, s_xyz) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    t_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="translateLocal",
    )
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))

    s_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeScale,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="scaleLocal",
    )
    s_op.Set(Gf.Vec3d(float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])))


def load_grasps_from_pkl(pkl_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    T = np.asarray(data["grasp_pose"], dtype=np.float64)
    width = np.asarray(data["width"], dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"grasp_pose should be (N,4,4), got {T.shape}")
    if width.ndim != 1 or width.shape[0] != T.shape[0]:
        raise ValueError(f"width should be (N,), got {width.shape}")
    return T, width


def semantic_rotation(approach: str, width: str, top: str) -> np.ndarray:
    R = np.column_stack([AXIS_VECS[approach], AXIS_VECS[width], AXIS_VECS[top]])
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(
            f"invalid right-handed assignment: {(approach, width, top)}, det={det:.1f}"
        )
    return R


def axis_name_from_vec(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=np.float64)
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if vec[idx] >= 0.0 else "-"
    base = ("x", "y", "z")[idx]
    unit = np.zeros(3, dtype=np.float64)
    unit[idx] = 1.0 if sign == "+" else -1.0
    if not np.allclose(vec, unit):
        raise ValueError(f"vector is not a signed basis axis: {vec}")
    return f"{sign}{base}"


def colored_axis(axis_name: str) -> str:
    base = axis_base(axis_name)
    return f"{ANSI[base]}{axis_name}{ANSI_RESET}"


@dataclass
class AxisAssignment:
    approach: str = "+x"
    width: str = "+y"
    top: str = "+z"

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.approach, self.width, self.top)


class GraspAxisViewer:
    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.assignment = AxisAssignment()
        self.running = True
        self.dirty = True
        self._keyboard_sub = None
        self._grasp_root_paths: list[str] = []
        self._base_transforms: np.ndarray | None = None

        self.T_batch, self.width_batch = load_grasps_from_pkl(args.pkl)
        count = self.T_batch.shape[0] if args.max_grasps <= 0 else min(self.T_batch.shape[0], args.max_grasps)
        self.T_batch = self.T_batch[:count]
        self.width_batch = self.width_batch[:count]

        rng = np.random.RandomState(args.seed)
        self.colors = rng.uniform(0.25, 0.90, (count, 3)).astype(np.float32)

        self._warm_up()
        self._load_asset()
        self._build_grasps_once()
        self._register_keyboard()
        self._print_help()
        self._apply_assignment()

    def _warm_up(self) -> None:
        for _ in range(30):
            simulation_app.update()

    def _load_asset(self) -> None:
        remove_prim_if_exists(self.stage, args.object_path)
        obj_prim = create_prim(args.object_path, prim_type="Xform")
        obj_prim.GetReferences().AddReference(args.usd)
        for _ in range(10):
            simulation_app.update()

    def _build_grasps_once(self) -> None:
        root_path = f"{args.object_path}/GraspDebug"
        remove_prim_if_exists(self.stage, root_path)
        create_prim(root_path, prim_type="Xform")

        self._base_transforms = self.T_batch.copy()
        self._grasp_root_paths = []
        for i, (T, width, color) in enumerate(zip(self.T_batch, self.width_batch, self.colors, strict=True)):
            grasp_path = f"{root_path}/grasp_{i:04d}"
            root_prim = create_prim(grasp_path, prim_type="Xform")
            self._grasp_root_paths.append(grasp_path)
            self._build_one_grasp(root_prim, float(width), color)
            set_local_matrix(root_prim, T)

        for _ in range(3):
            simulation_app.update()

    def _build_one_grasp(self, root_prim, width: float, color: np.ndarray) -> None:
        grasp_path = str(root_prim.GetPath())
        width = float(max(width, 1e-6))
        t = float(args.thickness)

        cube_specs = [
            ("handle", (-args.handle_len / 2.0, 0.0, 0.0), (args.handle_len, t, t)),
            ("bar", (0.0, 0.0, 0.0), (t, width, t)),
            ("finger_L", (args.finger_len / 2.0, -width / 2.0, 0.0), (args.finger_len, t, t)),
            ("finger_R", (args.finger_len / 2.0, width / 2.0, 0.0), (args.finger_len, t, t)),
        ]
        for name, translation, scale in cube_specs:
            prim = create_prim(f"{grasp_path}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, color)

        axis_specs = [
            ("x_axis", (args.axis_len / 2.0, 0.0, 0.0), (args.axis_len, args.axis_thickness, args.axis_thickness), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("y_axis", (0.0, args.axis_len / 2.0, 0.0), (args.axis_thickness, args.axis_len, args.axis_thickness), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("z_axis", (0.0, 0.0, args.axis_len / 2.0), (args.axis_thickness, args.axis_thickness, args.axis_len), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        axis_root = create_prim(f"{grasp_path}/axes", prim_type="Xform")
        for name, translation, scale, color_rgb in axis_specs:
            prim = create_prim(f"{axis_root.GetPath()}/{name}", prim_type="Cube")
            UsdGeom.Cube(prim).CreateSizeAttr().Set(1.0)
            set_local_translate_scale(prim, translation, scale)
            set_display_color(prim, color_rgb)

    def _is_valid_assignment(self, assignment: AxisAssignment) -> bool:
        bases = {axis_base(assignment.approach), axis_base(assignment.width), axis_base(assignment.top)}
        if len(bases) != 3:
            return False
        try:
            semantic_rotation(*assignment.as_tuple())
        except ValueError:
            return False
        return True

    def _completed_assignment(self, role: str, new_axis: str) -> AxisAssignment | None:
        candidate = AxisAssignment(
            approach=self.assignment.approach,
            width=self.assignment.width,
            top=self.assignment.top,
        )
        setattr(candidate, role, new_axis)

        if role in ("approach", "width"):
            if axis_base(candidate.approach) == axis_base(candidate.width):
                return None
            top_vec = np.cross(AXIS_VECS[candidate.approach], AXIS_VECS[candidate.width])
            candidate.top = axis_name_from_vec(top_vec)
            return candidate if self._is_valid_assignment(candidate) else None

        if role == "top":
            if axis_base(candidate.approach) == axis_base(candidate.top):
                return None
            width_vec = np.cross(AXIS_VECS[candidate.top], AXIS_VECS[candidate.approach])
            candidate.width = axis_name_from_vec(width_vec)
            return candidate if self._is_valid_assignment(candidate) else None

        return None

    def _cycle(self, role: str, delta: int) -> bool:
        current = getattr(self.assignment, role)
        start = AXES.index(current)
        for step in range(1, len(AXES) + 1):
            axis_name = AXES[(start + delta * step) % len(AXES)]
            candidate = self._completed_assignment(role, axis_name)
            if candidate is not None:
                self.assignment = candidate
                self.dirty = True
                self._print_state()
                return True
        return False

    def _flip(self, role: str) -> bool:
        current = getattr(self.assignment, role)
        flipped = ("-" if current[0] == "+" else "+") + current[-1]
        candidate = self._completed_assignment(role, flipped)
        if candidate is not None:
            self.assignment = candidate
            self.dirty = True
            self._print_state()
            return True
        return False

    def _reset_assignment(self) -> bool:
        self.assignment = AxisAssignment()
        self.dirty = True
        self._print_state()
        return True

    def _register_keyboard(self) -> None:
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        iface = carb.input.acquire_input_interface()
        key = carb.input.KeyboardInput

        def on_key(event, *args_unused):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return True
            if event.input in (key.A, key.LEFT):
                self._cycle("approach", -1)
            elif event.input in (key.D, key.RIGHT):
                self._cycle("approach", 1)
            elif event.input in (key.W, key.UP):
                self._cycle("width", -1)
            elif event.input in (key.S, key.DOWN):
                self._cycle("width", 1)
            elif event.input == key.Q:
                self._cycle("top", -1)
            elif event.input == key.E:
                self._cycle("top", 1)
            elif event.input == key.R:
                self._flip("approach")
            elif event.input == key.F:
                self._flip("width")
            elif event.input == key.T:
                self._flip("top")
            elif event.input == key.X:
                self._reset_assignment()
            elif event.input == key.H:
                self._print_help()
            elif event.input == key.ESCAPE:
                self.running = False
            return True

        self._keyboard_sub = iface.subscribe_to_keyboard_events(keyboard, on_key)

    def _apply_assignment(self) -> None:
        if self._base_transforms is None:
            return
        R = semantic_rotation(*self.assignment.as_tuple())
        semantic_T = np.eye(4, dtype=np.float64)
        semantic_T[:3, :3] = R

        for grasp_path, T in zip(self._grasp_root_paths, self._base_transforms, strict=True):
            prim = self.stage.GetPrimAtPath(grasp_path)
            set_local_matrix(prim, T @ semantic_T)

        for _ in range(2):
            simulation_app.update()
        self.dirty = False

    def _print_state(self) -> None:
        print(
            "  "
            f"approach = {colored_axis(self.assignment.approach)}  |  "
            f"width = {colored_axis(self.assignment.width)}  |  "
            f"top = {colored_axis(self.assignment.top)}"
        )

    def _print_help(self) -> bool:
        print("\n" + "=" * 76)
        print("Interactive Isaac Sim Grasp Axis Visualizer  |  grasp_viser_codex.py")
        print("=" * 76)
        print("A / D or Left / Right : cycle approach axis")
        print("W / S or Up / Down    : cycle width axis")
        print("Q / E                 : cycle top axis")
        print("R / F / T             : flip sign of approach / width / top")
        print("X                     : reset to (+x, +y, +z)")
        print("H                     : print help")
        print("Esc                   : quit")
        print("-" * 76)
        print("Canonical gripper convention before remapping:")
        print("  local +x = approach, local +y = width, local +z = top")
        print("  handle goes along -approach, fingers extend along +approach")
        print("  object USD stays visible under the same parent prim")
        print("-" * 76)
        print(
            f"Axis colors: {ANSI['x']}X{ANSI_RESET} red, "
            f"{ANSI['y']}Y{ANSI_RESET} green, "
            f"{ANSI['z']}Z{ANSI_RESET} blue"
        )
        print("Changed role is treated as the driver; the dependent axis auto-updates")
        print("to keep the frame orthogonal and right-handed.")
        print("=" * 76)
        self._print_state()
        return True

    def run(self) -> None:
        while simulation_app.is_running() and self.running:
            if self.dirty:
                self._apply_assignment()
            simulation_app.update()
        simulation_app.close()


def main() -> None:
    viewer = GraspAxisViewer()
    viewer.run()


if __name__ == "__main__":
    main()
