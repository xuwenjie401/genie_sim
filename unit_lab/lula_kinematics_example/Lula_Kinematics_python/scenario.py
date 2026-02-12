# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


class Robot:
    def __init__(self, name):
        self.name = name
        
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        if name == "Franka":
            self.robot_prim_path = "/panda"
            self.path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            self.robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
            self.urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
        elif name == "AgileX":
            self.robot_prim_path = "/aloha_description"
            # self.path_to_robot_usd = "/home/agxi/Documents/assets/robots/urdf_ws/src/aloha_new_description/urdf/aloha_new.usda"
            self.path_to_robot_usd = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/robot/curobo_robot/assets/robot/AgileX/aloha_new.usda"
            self.robot_description_path = "/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/AgileX/AgileX_fixed_left.yaml"
            self.urdf_path = "/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/AgileX/AgileX_aloha_fixed_dual.urdf"
        elif name == "ArmLeft":
            self.robot_prim_path = "/aloha_description"
            self.path_to_robot_usd = "/home/agxi/Documents/assets/robots/urdf_ws/src/aloha_new_description/urdf/sl_arm_left/sl_arm_left.usd"
            self.robot_description_path = "/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/AgileX/arm_left_only.yaml"
            self.urdf_path = "/home/agxi/Documents/assets/robots/urdf_ws/src/aloha_new_description/urdf/sl_arm_left.urdf"


class FrankaKinematicsExample:
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None

        # self.robot = Robot("Franka")
        self.robot = Robot("AgileX")
        # self.robot = Robot("ArmLeft")

    def load_example_assets(self):
        # Add the Robot and target to the stage

        robot_prim_path = self.robot.robot_prim_path
        path_to_robot_usd = self.robot.path_to_robot_usd

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self._target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        if self.robot.name == "AgileX":
            self._target.set_default_state(np.array([0.5, 0.3, 1.0]), euler_angles_to_quats([np.pi/2.0, np.pi/2.0, 0.0]))
        elif self.robot.name == "ArmLeft":
            # self._target.set_default_state(np.array([0.5, 0.0, 0.2]), euler_angles_to_quats([np.pi/2.0, np.pi/3.0, -np.pi/2.0]))
            self._target.set_default_state(np.array([0.4, -0.1, 0.1]), euler_angles_to_quats([np.pi/2.0, 75.0/180.0*np.pi, 0.0]))

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=self.robot.robot_description_path,
            urdf_path=self.robot.urdf_path,
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "right_gripper" 
        if self.robot.name == "AgileX" or self.robot.name == "ArmLeft":
            end_effector_name = "fl_ee_link"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, step: float):
        target_position, target_orientation = self._target.get_world_pose()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken")

        # Unused Forward Kinematics:
        # ee_position,ee_rot_mat = articulation_kinematics_solver.compute_end_effector_pose()

    def reset(self):
        # Kinematics is stateless
        pass
