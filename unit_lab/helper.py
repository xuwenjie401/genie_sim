#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, List

# Third Party
import numpy as np
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from pxr import UsdPhysics

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform

ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

try:
    # import for older isaacsim installations
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    # import for isaac sim 4.5+
    from isaacsim.core.api.materials import OmniPBR


# Standard Library
from typing import Optional

# Third Party
from omni.isaac.core.utils.extensions import enable_extension

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = True,
    # load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
    initialize_world: bool = True,
):
    stage = my_world.stage

    if load_from_usd:

        from isaacsim.core.utils.stage import add_reference_to_stage

        # 获取配置中的 USD 路径和机器人根节点
        usd_path = robot_config["kinematics"]["usd_path"]
        usd_robot_root = robot_config["kinematics"].get("usd_robot_root", "")
        base_link_name = robot_config["kinematics"]["base_link"]

        # 定义 prim 路径（根节点位置）
        prim_path = "/World"

        # 确保 usd_robot_root 在正确的位置拼接
        if usd_robot_root:
            if not usd_robot_root.startswith("/"):
                usd_robot_root = "/" + usd_robot_root  # 确保以斜杠开始
            robot_root_path = prim_path + usd_robot_root  # 拼接路径
        else:
            robot_root_path = prim_path  # 如果没有设置 usd_robot_root，直接用 prim_path

        add_reference_to_stage(usd_path, robot_root_path)

        # 检查 prim 是否已经在场景中创建
        # link_prim = stage.GetPrimAtPath(robot_root_path + "/" + base_link_name)
        # breakpoint()
        # if not link_prim:
        #     raise Exception(f"Prim at path {robot_root_path}/{base_link_name} not found.")

        from pxr import UsdPhysics

        # 获取该 Prim
        prim = stage.GetPrimAtPath(robot_root_path)
        # 如果它没定义 ArticulationRoot，我们手动给它加上
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(prim)

        # 添加机器人到场景
        robot_p = Robot(
            prim_path=robot_root_path + "/" + base_link_name,
            # prim_path=robot_root_path,
            name=robot_name,
        )

        robot_prim = robot_p.prim
        stage = robot_prim.GetStage()
        linkp = stage.GetPrimAtPath(robot_root_path)
        set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

        robot = my_world.scene.add(robot_p)

        # 初始化物理世界
        if initialize_world:
            my_world.initialize_physics()
            robot.initialize()

        return robot, robot_root_path

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )

        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:
        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    return robot, robot_path


class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))
