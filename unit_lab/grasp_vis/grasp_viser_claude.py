"""
grasp_viser_claude.py  —  Isaac Sim interactive grasp axis visualizer

Shows the 3D USD asset + C-shaped gripper overlays.
Press keys to instantly cycle approach / width axes; grasps respawn each change.

Controls  (Isaac Sim viewport must have focus)
──────────────────────────────────────────────
  ← →  (or A D)  :  cycle APPROACH axis  (+x -x +y -y +z -z)
  ↑ ↓  (or W S)  :  cycle WIDTH / grip axis  (skips same base-axis as approach)
  H               :  reprint help
  Q               :  quit

Axis semantics
  approach  =  finger direction (fingers extend in +dir, handle in -dir)
  width     =  opening direction (fingers placed at ±w/2)
  top       =  palm normal (auto = remaining axis)
"""

import pickle
import numpy as np

import isaacsim  # noqa: F401 — must be first import

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--pkl",        type=str,   default=(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/"
    "GenieSimAssets/interaction/benchmark_beverage_bottle_008/"
    "grasp_pose/grasp_pose.pkl"))
_parser.add_argument("--usd",        type=str,   default=(
    "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/"
    "GenieSimAssets/objects/benchmark/beverage_bottle/"
    "benchmark_beverage_bottle_008/Aligned.usda"))
_parser.add_argument("--max_grasps", type=int,   default=5)
_parser.add_argument("--finger_len", type=float, default=0.04)
_parser.add_argument("--handle_len", type=float, default=0.08)
_parser.add_argument("--thickness",  type=float, default=0.003)
_parser.add_argument("--seed",       type=int,   default=0)
_args, _ = _parser.parse_known_args()

# ── Isaac Sim must be booted before any omni/pxr imports ─────────────────────
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"width": 1920, "height": 1080, "headless": False})

import carb
import carb.input
import omni.usd
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.prims import create_prim

# ─── Axis definitions ─────────────────────────────────────────────────────────
AXES = ["+x", "-x", "+y", "-y", "+z", "-z"]
_BIDX = {"x": 0, "y": 1, "z": 2}

_ANSI = {"x": "\033[31m", "y": "\033[32m", "z": "\033[34m"}
_RST  = "\033[0m"


# ─── USD utility functions (same convention as original grasp_viser.py) ───────

def _np_to_gf(T: np.ndarray) -> Gf.Matrix4d:
    M = np.asarray(T, dtype=np.float64).T
    return Gf.Matrix4d(
        M[0,0], M[0,1], M[0,2], M[0,3],
        M[1,0], M[1,1], M[1,2], M[1,3],
        M[2,0], M[2,1], M[2,2], M[2,3],
        M[3,0], M[3,1], M[3,2], M[3,3],
    )


def _set_transform(prim, T: np.ndarray):
    xf = UsdGeom.Xformable(prim)
    op = None
    for o in xf.GetOrderedXformOps():
        if o.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op = o; break
    if op is None:
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(_np_to_gf(T))


def _set_ts(prim, t, s):
    """Set translate + scale on a fresh xformOp stack."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    t_op = xf.AddXformOp(UsdGeom.XformOp.TypeTranslate,
                          UsdGeom.XformOp.PrecisionDouble, opSuffix="t")
    t_op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    s_op = xf.AddXformOp(UsdGeom.XformOp.TypeScale,
                          UsdGeom.XformOp.PrecisionDouble, opSuffix="s")
    s_op.Set(Gf.Vec3d(float(s[0]), float(s[1]), float(s[2])))


def _set_color(prim, rgb):
    c = np.asarray(rgb, dtype=np.float32)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set(
        [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))])


def _remove(stage, path: str):
    p = stage.GetPrimAtPath(path)
    if p and p.IsValid():
        stage.RemovePrim(path)


# ─── Grasp spawn ──────────────────────────────────────────────────────────────

def _parse_axis(ax: str):
    s = ax.strip().lower()
    sign = -1.0 if s.startswith("-") else 1.0
    name = s.lstrip("+-")
    v = np.zeros(3, dtype=np.float64)
    v[_BIDX[name]] = sign
    return name, _BIDX[name], v


def spawn_grasps(
    stage,
    T_batch: np.ndarray,
    width: np.ndarray,
    parent_path: str,
    direction_axis: str,
    grip_axis: str,
    finger_len: float,
    handle_len: float,
    thickness: float,
    seed: int = 0,
):
    dn, di, dv = _parse_axis(direction_axis)
    gn, gi, gv = _parse_axis(grip_axis)
    kn = ({"x", "y", "z"} - {dn, gn}).pop()
    ki = _BIDX[kn]
    kv = np.zeros(3, dtype=np.float64); kv[ki] = 1.0

    N = T_batch.shape[0]
    rng = np.random.RandomState(seed)
    colors = rng.uniform(0.3, 0.9, (N, 3)).astype(np.float32)

    def tv(d=0., g=0., k=0.):
        v = d*dv + g*gv + k*kv
        return (float(v[0]), float(v[1]), float(v[2]))

    def sv(d=0., g=0., k=0.):
        s = np.zeros(3)
        s[di] = abs(d); s[gi] = abs(g); s[ki] = abs(k)
        return (float(s[0]), float(s[1]), float(s[2]))

    root = f"{parent_path}/Grasps"
    _remove(stage, root)
    create_prim(root, prim_type="Xform")

    for i in range(N):
        gp = f"{root}/g{i:04d}"
        xf = create_prim(gp, prim_type="Xform")
        _set_transform(xf, T_batch[i])

        t = thickness
        w = float(max(width[i], 1e-6))
        c = colors[i]

        # handle
        h = create_prim(f"{gp}/handle", prim_type="Cube")
        UsdGeom.Cube(h).CreateSizeAttr().Set(1.0)
        _set_ts(h, tv(d=-handle_len/2), sv(d=handle_len, g=t, k=t))
        _set_color(h, c)

        # crossbar
        cb = create_prim(f"{gp}/bar", prim_type="Cube")
        UsdGeom.Cube(cb).CreateSizeAttr().Set(1.0)
        _set_ts(cb, tv(), sv(d=t, g=w, k=t))
        _set_color(cb, c)

        # left finger
        fl = create_prim(f"{gp}/fL", prim_type="Cube")
        UsdGeom.Cube(fl).CreateSizeAttr().Set(1.0)
        _set_ts(fl, tv(d=+finger_len/2, g=-w/2), sv(d=finger_len, g=t, k=t))
        _set_color(fl, c)

        # right finger
        fr = create_prim(f"{gp}/fR", prim_type="Cube")
        UsdGeom.Cube(fr).CreateSizeAttr().Set(1.0)
        _set_ts(fr, tv(d=+finger_len/2, g=+w/2), sv(d=finger_len, g=t, k=t))
        _set_color(fr, c)


def spawn_axes(
    stage,
    T_batch: np.ndarray,
    parent_path: str,
    axis_len: float = 0.06,
    axis_thick: float = 0.002,
):
    root = f"{parent_path}/Axes"
    _remove(stage, root)
    create_prim(root, prim_type="Xform")

    cols = {"x": [1,0,0], "y": [0,1,0], "z": [0,0,1]}
    N = T_batch.shape[0]

    for i in range(N):
        ap = f"{root}/g{i:04d}"
        axf = create_prim(ap, prim_type="Xform")
        _set_transform(axf, T_batch[i])

        for name in ("x", "y", "z"):
            ai = _BIDX[name]
            t = [0., 0., 0.]; t[ai] = axis_len / 2
            s = [axis_thick]*3; s[ai] = axis_len
            p = create_prim(f"{ap}/{name}", prim_type="Cube")
            UsdGeom.Cube(p).CreateSizeAttr().Set(1.0)
            _set_ts(p, t, s)
            _set_color(p, cols[name])


# ─── State + keyboard ─────────────────────────────────────────────────────────

class _State:
    dir_idx  = 0   # → "+x"
    grip_idx = 2   # → "+y"
    dirty    = True
    running  = True

    def base(self, idx): return AXES[idx].lstrip("+-")

    def fix_grip(self):
        if self.base(self.grip_idx) == self.base(self.dir_idx):
            self._step_grip(+1)

    def _step_grip(self, delta):
        n = len(AXES)
        for _ in range(n):
            self.grip_idx = (self.grip_idx + delta) % n
            if self.base(self.grip_idx) != self.base(self.dir_idx):
                break

    def step_dir(self, delta):
        self.dir_idx = (self.dir_idx + delta) % len(AXES)
        self.fix_grip()
        self.dirty = True
        _print_state(self)

    def step_grip(self, delta):
        self._step_grip(delta)
        self.dirty = True
        _print_state(self)


def _print_state(s: _State):
    d  = AXES[s.dir_idx]
    g  = AXES[s.grip_idx]
    db = d.lstrip("+-"); gb = g.lstrip("+-")
    top = ({"x","y","z"} - {db, gb}).pop()
    dc, gc, tc = _ANSI.get(db,""), _ANSI.get(gb,""), _ANSI.get(top,"")
    print(f"  approach = {dc}{d:3s}{_RST}  │  width = {gc}{g:3s}{_RST}  │  top = {tc}+{top}{_RST}")


def _print_help():
    print("\n" + "─"*56)
    print("  INTERACTIVE GRASP VISUALIZER  (grasp_viser_claude.py)")
    print("─"*56)
    print("  ← →  (A D)   cycle approach axis  (finger direction)")
    print("  ↑ ↓  (W S)   cycle grip / width axis")
    print("  H             reprint this help")
    print("  Q             quit")
    print("─"*56)
    print(f"  {_ANSI['x']}RED{_RST}=X  {_ANSI['g' if False else 'y']}GREEN{_RST}=Y  {_ANSI['z']}BLUE{_RST}=Z  (coord axes overlay)")
    print("  approach — fingers extend in +dir; handle points in -dir")
    print("  width    — crossbar / finger spread along this axis")
    print("  top      — palm normal (implicit, remaining axis)")
    print("─"*56)
    print("  ⚠  Isaac Sim viewport must have focus for keys to work")
    print("─"*56)


def setup_keyboard(state: _State):
    import omni.appwindow
    iface = carb.input.acquire_input_interface()
    kb    = omni.appwindow.get_default_app_window().get_keyboard()
    ki    = carb.input.KeyboardInput

    def on_key(event, *_, **__):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        k = event.input
        if   k in (ki.LEFT,  ki.A): state.step_dir(-1)
        elif k in (ki.RIGHT, ki.D): state.step_dir(+1)
        elif k in (ki.DOWN,  ki.S): state.step_grip(-1)
        elif k in (ki.UP,    ki.W): state.step_grip(+1)
        elif k == ki.H:             _print_help()
        elif k == ki.Q:             state.running = False
        return True

    # Keep sub alive — must not be GC'd
    sub = iface.subscribe_to_keyboard_events(kb, on_key)
    return sub


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load grasps once
    with open(_args.pkl, "rb") as f:
        data = pickle.load(f)
    T_all = np.asarray(data["grasp_pose"], dtype=np.float64)
    W_all = np.asarray(data["width"],      dtype=np.float64)
    N = min(T_all.shape[0], _args.max_grasps) if _args.max_grasps > 0 else T_all.shape[0]
    T = T_all[:N]
    W = W_all[:N]
    print(f"Loaded {T_all.shape[0]} grasps — showing {N}")

    # Warm-up
    for _ in range(30):
        simulation_app.update()

    # Load object USD
    stage = omni.usd.get_context().get_stage()
    obj_path = "/World/Object"
    _remove(stage, obj_path)
    obj_prim = create_prim(obj_path, prim_type="Xform")
    obj_prim.GetReferences().AddReference(_args.usd)
    print(f"Loaded USD: {_args.usd}")

    for _ in range(10):
        simulation_app.update()

    state = _State()
    _kb_sub = setup_keyboard(state)  # keep reference alive

    _print_help()

    def respawn():
        d_ax = AXES[state.dir_idx]
        g_ax = AXES[state.grip_idx]
        spawn_grasps(
            stage, T, W,
            parent_path=obj_path,
            direction_axis=d_ax,
            grip_axis=g_ax,
            finger_len=_args.finger_len,
            handle_len=_args.handle_len,
            thickness=_args.thickness,
            seed=_args.seed,
        )
        spawn_axes(stage, T, parent_path=obj_path)
        for _ in range(3):
            simulation_app.update()

    # Render loop
    while simulation_app.is_running() and state.running:
        if state.dirty:
            state.dirty = False
            respawn()
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
