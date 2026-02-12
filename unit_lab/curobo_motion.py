import copy
import re
import time
from typing import Optional

import carb
import numpy as np
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
print(f"check root dir: {root_dir}")
sys.path.append(root_dir)

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper, get_prim_world_pose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenPlanConfig, PoseCostMetric
from isaacsim.core.api import World
from isaacsim.core.api.objects import sphere
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.types import ArticulationAction

from source.data_collection.server.motion_generator.mesh_utils import get_mesh_attrs, simplify_obstacles_from_stage
from source.data_collection.common.base_utils.logger import logger


# USD imports (using try-except for optional dependency)
try:
    from pxr import UsdGeom, UsdPhysics
except ImportError:
    # These will be imported dynamically when needed
    pass


CUROBO_BATCH_SIZE = 2
MAX_MESH_FACES = 1000


class CuroboUsdHelper(UsdHelper):
    def get_obstacles_from_stage(
        self, 
        only_paths: Optional[list] = None,
        ignore_paths: Optional[list] = None,
        only_substring: Optional[list] = None,
        ignore_substring: Optional[list] = None,
        reference_prim_path: Optional[str] = None,
        timecode: float = 0,
    ) -> WorldConfig:
        obstacles = {
            "cuboid": None,
            "sphere": None,
            "mesh": None,
            "cylinder": None,
            "capsule": None,
        }

        r_T_w = None
        # use the instance xform cache
        try:
            self._xform_cache.Clear()
            self._xform_cache.SetTime(timecode)
        except Exception:
            # fallback: create local cache
            try:
                self._xform_cache = UsdGeom.XformCache(timecode)
            except Exception:
                pass

        if reference_prim_path is not None:
            reference_prim = self.stage.GetPrimAtPath(reference_prim_path)
            r_T_w, _ = get_prim_world_pose(self._xform_cache, reference_prim, inverse=True)

        # iterate stage prims (use Traverse for full traversal)
        all_items = self.stage.Traverse()
        for prim in all_items:
            prim_path = str(prim.GetPath())

            # only/ignore path filters
            if only_paths is not None and not any([prim_path.startswith(k) for k in only_paths]):
                continue
            if ignore_paths is not None and any([prim_path.startswith(k) for k in ignore_paths]):
                continue
            if only_substring is not None and not any([k in prim_path for k in only_substring]):
                continue
            if ignore_substring is not None and any([k in prim_path for k in ignore_substring]):
                continue

            # Optionally check for collision enabled attribute
            try:
                collisionAPI = UsdPhysics.CollisionAPI.Get(self.stage, prim_path)
                if collisionAPI and not collisionAPI.GetCollisionEnabledAttr().Get():
                    # skip prims that explicitly disable collision
                    continue
            except Exception:
                # if we can't query collision API, proceed normally
                pass

            try:
                if prim.IsA(UsdGeom.Mesh):
                    if obstacles["mesh"] is None:
                        obstacles["mesh"] = []
                    # use the local get_mesh_attrs (triangulating wrapper)
                    m_data = get_mesh_attrs(prim, cache=self._xform_cache, transform=r_T_w)
                    if m_data is not None:
                        obstacles["mesh"].append(m_data)
            except Exception as e:
                logger.error(f"Error extracting prim {prim_path}: {e}")
                continue

        world_model = WorldConfig(**obstacles)
        return world_model


class CuroboMotion:
    world_coll_checker = None
    cached_obstacle_info = {}
    curobo_kinematics = None
    curobo_kinematics_robot_cfg = {}

    def __init__(
        self,
        name: str,
        robot: Articulation,
        world: World,
        robot_cfg,
        robot_prim_path,
        robot_list,
        step=100,
        debug=False,
    ):
        self.name = name
        self.debug = debug
        self.usd_help = CuroboUsdHelper()
        self.target_pose = None
        self.target_orientation = None
        self.past_pose = None
        self.past_orientation = None
        self.robot_list = robot_list
        tensor_args = TensorDeviceType()
        self.robot_prim_path = robot_prim_path

        n_obstacle_cuboids = 30
        self.init_ee_pose = {}
        n_obstacle_mesh = 30

        robot_cfg_path = get_robot_configs_path()
        self.robot_cfg = load_yaml(join_path(robot_cfg_path, robot_cfg))['robot_cfg']

        # attached object  (omitted)

        self.lock_joints = self.robot_cfg["kinematics"]["lock_joints"]
        self.lock_js_names = []
        if self.lock_joints:
            for key in self.lock_joints:
                self.lock_js_names.append(key)
        j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]

        self.world_cfg = WorldConfig()
        motion_gen_config = MotionGen.load_from_robot_config(
            robot_cfg=self.robot_cfg,
            world_model=self.world_cfg,
            tensor_args=tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_trajopt_seeds=4,
            num_graph_seeds=4,
            num_ik_seeds=32,
            num_batch_ik_seeds=32,
            interpolation_dt=0.01,
            interpolation_steps=5000,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=True,
            trajopt_dt=None,
            trajopt_tsteps=step,
            num_trajopt_noisy_seeds=1,
            num_batch_trajopt_seeds=1,
            collision_activation_distance=0.01,
            world_coll_checker=CuroboMotion.world_coll_checker,
        )

        self.tensor_args = tensor_args
        self.motion_gen = MotionGen(motion_gen_config)

        if CuroboMotion.world_coll_checker is None:
            CuroboMotion.world_coll_checker = self.motion_gen.world_coll_checker

        self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        self.world_model = self.motion_gen.world_collision
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
        )


    def reset(self):
        self.motion_gen.clear_world_cache()
