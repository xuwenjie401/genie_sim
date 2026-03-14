"""
Calibration script: measure galbot torso_base_link world-Z vs. leg_joint1/2/3 delta.

Usage:
    cd unit_lab
    python calibrate_leg_height.py

Output:
    A table of (delta, j1, j2, j3, torso_z, height_change) printed to stdout.
    Use this to calibrate GALBOT_HEIGHT_PER_LEG_DELTA and verify sign convention
    in free_worker_vertical.py.

Constraint: joint1 + joint3 = joint2  (keeps torso approximately vertical)
Parameterization:
    joint1 = J1_DEFAULT + delta
    joint2 = J2_DEFAULT + 2 * delta
    joint3 = J3_DEFAULT + delta
Negative delta → joints decrease → check whether torso goes UP or DOWN.
"""

try:
    import isaacsim
except ImportError:
    pass

import torch
a = torch.zeros(4, device="cuda:0")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--physics_step", type=int, default=60)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1920",
    "height": "1080",
})
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

import numpy as np
import time
import os
import sys

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage

try:
    from pxr import Gf, Sdf, UsdPhysics
except ImportError:
    pass

import omni

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    sys.path.append(os.path.join(root_dir, "source/data_collection"))

# ── Config ──────────────────────────────────────────────────────────────────
ROBOT_USD = "/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_fixed.usda"
SCENE_USD  = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_00.usda"
ROBOT_PRIM = "/galbot_one_golf"
TORSO_PRIM = "/galbot_one_golf/torso_base_link"

# Default reference leg joint values (from basic_test.yaml retract_config)
J1_DEFAULT = 0.5236
J2_DEFAULT = 1.0821
J3_DEFAULT = 0.6109

# Settle steps: how many physics steps to wait after setting joints
SETTLE_STEPS = 120  # 2 s at 60 Hz

# Delta range to sweep
DELTA_VALUES = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
# ────────────────────────────────────────────────────────────────────────────


def main():
    physics_dt   = 1.0 / args.physics_step
    rendering_dt = 1.0 / 30.0

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=physics_dt,
        rendering_dt=rendering_dt,
        device="cpu",
    )

    # Load robot
    add_reference_to_stage(ROBOT_USD, ROBOT_PRIM)
    add_reference_to_stage(SCENE_USD, "/World")

    # Physics scene
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    world.play()

    # Init articulation
    articulation = Articulation(prim_path=ROBOT_PRIM, name="galbot")
    world.scene.add(articulation)
    world.reset()
    articulation.initialize()

    # Get joint indices
    j1_idx = articulation.get_dof_index("leg_joint1")
    j2_idx = articulation.get_dof_index("leg_joint2")
    j3_idx = articulation.get_dof_index("leg_joint3")
    print(f"leg_joint1 idx={j1_idx}, leg_joint2 idx={j2_idx}, leg_joint3 idx={j3_idx}")

    # Set default joints and settle
    articulation.set_joint_positions(
        np.array([J1_DEFAULT, J2_DEFAULT, J3_DEFAULT]),
        joint_indices=np.array([j1_idx, j2_idx, j3_idx]),
    )
    for _ in range(SETTLE_STEPS):
        world.step(render=False)

    # Record baseline torso Z at delta=0 (may differ from DEFAULT if USD has different pose)
    torso_xform = XFormPrim(TORSO_PRIM)
    baseline_pos, _ = torso_xform.get_world_pose()
    baseline_z = float(baseline_pos[2])
    print(f"\nBaseline torso_base_link Z at default joints: {baseline_z:.4f} m")
    print(f"Default joints: j1={J1_DEFAULT:.4f}, j2={J2_DEFAULT:.4f}, j3={J3_DEFAULT:.4f}")
    print(f"Constraint check: j1+j3={J1_DEFAULT+J3_DEFAULT:.4f} vs j2={J2_DEFAULT:.4f}")
    print()

    header = f"{'delta':>7}  {'j1':>7}  {'j2':>7}  {'j3':>7}  {'torso_z':>9}  {'delta_z':>9}  {'m/rad':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for delta in DELTA_VALUES:
        j1 = J1_DEFAULT + delta
        j2 = J2_DEFAULT + 2.0 * delta
        j3 = J3_DEFAULT + delta

        articulation.set_joint_positions(
            np.array([j1, j2, j3]),
            joint_indices=np.array([j1_idx, j2_idx, j3_idx]),
        )

        for _ in range(SETTLE_STEPS):
            world.step(render=False)

        pos, _ = torso_xform.get_world_pose()
        torso_z = float(pos[2])
        delta_z = torso_z - baseline_z
        m_per_rad = (delta_z / delta) if delta != 0.0 else float("nan")

        results.append((delta, j1, j2, j3, torso_z, delta_z, m_per_rad))
        print(f"{delta:>7.3f}  {j1:>7.4f}  {j2:>7.4f}  {j3:>7.4f}  {torso_z:>9.4f}  {delta_z:>+9.4f}  {m_per_rad:>9.4f}")

    # Summary
    print()
    print("=" * 60)
    finite = [(d, dz, mpr) for d, *_, dz, mpr in results if abs(d) > 1e-6]
    if finite:
        avg_mpr = np.mean([mpr for _, _, mpr in finite])
        print(f"Average torso Z change per radian of delta: {avg_mpr:.4f} m/rad")
        print(f"Sign: {'NEGATIVE delta lowers torso' if avg_mpr > 0 else 'POSITIVE delta lowers torso'}")
        print()
        print("Use in free_worker_vertical.py:")
        print(f"  GALBOT_HEIGHT_PER_LEG_DELTA = {abs(avg_mpr):.4f}  # m/rad")
        if avg_mpr > 0:
            print("  Sign: subtract delta to lower  (joint_new = joint_default - delta)")
        else:
            print("  Sign: add delta to lower       (joint_new = joint_default + delta)")

    simulation_app.close()


if __name__ == "__main__":
    main()
