try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
import time

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--physics_step",
    type=int,
    default=60,
)

args = parser.parse_args()

############################################################

# Third Party
# from omni.isaac.kit import SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)

# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene

from isaacsim.core.api import World
from isaacsim.core.api.objects import cuboid, sphere
from isaacsim.core.utils import extensions
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
from isaacsim.core.utils.stage import add_reference_to_stage

from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleGeometryPrim as GeometryPrim
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim

import omni
import asyncio
import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


ROBOT_USD_FILE = "/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_fixed.usda"
# ROBOT_USD_FILE = "/home/agxi/Documents/assets/robots/urdf_ws/src/galbot_one_golf_description/sim_ready/galbot_one_golf/usdv2/galbot_root.usda"
# ROBOT_USD_FILE = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/robot/G2_omnipicker/robot_fix.usda"

SCENE_USD_FILE = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_00.usda"

class BasicRunner:
    def __init__(self):

        physics_dt = (float)(1 / args.physics_step)
        rendering_dt = (float)(1 / 30)

        # 创建仿真世界
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            device="cpu",
        )
        self.stage = self.world.stage

        # 地面
        # self.world.scene.add_default_ground_plane()

        robot_prim_path = "/galbot_one_golf"
        init_position = [2.15, 0.7757971635415469, 0.0]
        init_rotation = [1, 0, 0, 0]

        add_reference_to_stage(ROBOT_USD_FILE, robot_prim_path)
        add_reference_to_stage(SCENE_USD_FILE, "/World")

        self.usd_robot = XFormPrim(
            prim_path=robot_prim_path,
            position=init_position,
            orientation=init_rotation
        )

        self.world.play()


    def run(self):
        
        while simulation_app.is_running():

            self.world.step(render=False)
            self.world.render()

        simulation_app.close()


if __name__ == "__main__":
    runner = BasicRunner()
    runner.run()

