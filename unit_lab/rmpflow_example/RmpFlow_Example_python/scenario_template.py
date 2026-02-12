# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)


class FrankaRmpFlowExample:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._dbg_mode = False

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])

        self._obstacle = FixedCuboid(
            "/World/obstacle", size=0.05, position=np.array([0.4, 0.0, 0.65]), color=np.array([0.0, 0.0, 1.0])
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._obstacle

    def setup(self):
        # Loading RMPflow can be done quickly for supported robots
        print("Supported Robots with a Provided RMPflow Config:", list(get_supported_robot_policy_pairs().keys()))
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)
        self._rmpflow.add_obstacle(self._obstacle)

        if self._dbg_mode:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0] / 50
            self._articulation.get_articulation_controller().set_gains(kps=bad_proportional_gains)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(target_position, target_orientation)

        # Track any movements of the cube obstacle
        self._rmpflow.update_world()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be
        if self._dbg_mode:
            # RMPflow was set to roll out robot state internally, assuming that all returned
            # joint targets were hit exactly.
            self._rmpflow.reset()
            self._rmpflow.visualize_collision_spheres()

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))
