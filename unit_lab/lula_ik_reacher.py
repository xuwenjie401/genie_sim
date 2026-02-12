"""
Interactive Lula IK demo (Isaac Sim)

- Creates a draggable target cube at /World/target
- Loads a robot articulation into the stage
- Uses LulaKinematicsSolver + ArticulationKinematicsSolver to solve IK each time target moves
- Applies the returned ArticulationAction to the robot

Usage example (GUI):
    ./isaac-sim.sh -p this_script.py -- \
        --robot_usd /absolute/or/nucleus/path/to/robot.usd \
        --robot_prim_path /World/Robot \
        --robot_description_yaml /absolute/path/to/robot_description.yaml \
        --urdf_path /absolute/path/to/robot.urdf \
        --ee_link fl_ee_link

Headless:
    ./python.sh this_script.py --headless
"""

import argparse
import numpy as np

# Isaac Sim
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run headless", default=False)

# Robot loading
parser.add_argument("--robot_usd", type=str, required=True, help="USD file path for robot")
parser.add_argument("--robot_prim_path", type=str, default="/World/Robot", help="Prim path to spawn robot under")

# Lula IK inputs
parser.add_argument("--robot_description_yaml", type=str, required=True, help="Lula robot_description.yaml path")
parser.add_argument("--urdf_path", type=str, required=True, help="URDF path used by Lula")
parser.add_argument("--ee_link", type=str, required=True, help="End-effector link name (must match Lula description)")

# IK tolerances (you can loosen for debugging)
parser.add_argument("--pos_tol", type=float, default=1e-3, help="Position tolerance (meters)")
parser.add_argument("--rot_tol", type=float, default=5e-2, help="Orientation tolerance (radians-ish)")

# Target cube
parser.add_argument("--target_xyz", nargs=3, type=float, default=[0.5, 0.0, 0.5], help="Initial target position")
parser.add_argument("--target_size", type=float, default=0.05, help="Target cube size")

args = parser.parse_args()

# Start Isaac
simulation_app = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

# Isaac Sim core
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation

# Lula
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver


def main():
    # Create a world
    world = World(stage_units_in_meters=1.0)
    stage = get_current_stage()

    # Ensure /World exists and is default prim (optional, but helps with consistent paths)
    # If your stage already has /World as default, this is fine.
    if not stage.GetPrimAtPath("/World"):
        stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Add a ground plane for visual reference
    world.scene.add_default_ground_plane()

    # Spawn a draggable target cube (you can drag in viewport when not headless)
    target = cuboid.VisualCuboid(
        prim_path="/World/target",
        position=np.array(args.target_xyz, dtype=np.float32),
        orientation=np.array([0, 1, 0, 0], dtype=np.float32),  # (w,x,y,z) in Isaac core objects
        color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        size=float(args.target_size),
    )

    # Load your robot USD
    # - This assumes the USD contains an articulation (joints etc).
    add_reference_to_stage(args.robot_usd, args.robot_prim_path)

    # Wrap as an Articulation so we can read pose and apply actions
    robot = Articulation(args.robot_prim_path)
    world.scene.add(robot)

    # Build Lula solver + articulation kinematics solver
    # IMPORTANT:
    # - robot_description_yaml and urdf_path must be consistent with your robot model
    lula_solver = LulaKinematicsSolver(
        robot_description_path=args.robot_description_yaml,
        urdf_path=args.urdf_path,
    )
    art_ik_solver = ArticulationKinematicsSolver(
        robot, lula_solver, args.ee_link
    )

    articulation_controller = None

    # Track last target pose to avoid spamming IK every frame
    last_target_p = None
    last_target_q = None

    # Main sim loop
    while simulation_app.is_running():
        world.step(render=True)

        # Wait for user to click Play
        if not world.is_playing():
            continue

        # Create controller lazily (after sim starts)
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()

        # Read current target pose
        tgt_p, tgt_q = target.get_world_pose()  # position (xyz), orientation quaternion
        tgt_p = np.array(tgt_p, dtype=np.float32)
        tgt_q = np.array(tgt_q, dtype=np.float32)

        # Only solve when the target changes (dragging)
        if last_target_p is not None and last_target_q is not None:
            if np.linalg.norm(tgt_p - last_target_p) < 1e-6 and np.linalg.norm(tgt_q - last_target_q) < 1e-6:
                continue

        # Update last pose
        last_target_p = tgt_p.copy()
        last_target_q = tgt_q.copy()

        # Set robot base pose in Lula (critical if robot moves in world)
        base_p, base_q = robot.get_world_pose()
        lula_solver.set_robot_base_pose(base_p, base_q)

        # Compute IK
        # Notes:
        # - compute_inverse_kinematics expects position + orientation
        # - Some builds accept tolerance kwargs; if yours doesn't, remove them
        try:
            actions, success = art_ik_solver.compute_inverse_kinematics(
                target_position=tgt_p,
                target_orientation=tgt_q,
                position_tolerance=args.pos_tol,
                orientation_tolerance=args.rot_tol,
            )
        except TypeError:
            # Older/newer API variants may not accept tolerance kwargs
            actions, success = art_ik_solver.compute_inverse_kinematics(
                target_position=tgt_p,
                target_orientation=tgt_q,
            )

        # Apply if success
        if success:
            # `actions` is usually an ArticulationAction already, but some versions return raw arrays.
            if isinstance(actions, ArticulationAction):
                articulation_controller.apply_action(actions)
            else:
                # Fallback: assume `actions` is a joint position array in robot DOF order.
                articulation_controller.apply_action(ArticulationAction(joint_positions=actions))
        else:
            # Minimal feedback in console (you can add carb.log_warn if you want)
            print("[LULA IK] failed (success=False). Try loosening tolerances or verifying base_link/ee_link mapping.")

    simulation_app.close()


if __name__ == "__main__":
    main()
