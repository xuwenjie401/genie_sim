# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os

import omni.kit.commands
import omni.usd
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from pxr import Gf, Sdf, UsdLux, UsdShade

import numpy as np
from scipy.spatial.transform import Rotation


def batch_matrices_to_quaternions_scipy(pose_matrices):
    """
    Convert batch 4x4 pose matrices to quaternions using Scipy's Rotation library

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [x, y, z, w] (Scipy default order)
    """
    # Extract rotation matrix part (n, 3, 3)
    rotation_matrices = pose_matrices[:, :3, :3]

    # Create Rotation object and convert to quaternion
    rot = Rotation.from_matrix(rotation_matrices)
    quaternions = rot.as_quat()  # Returns (n, 4) array, order is [x, y, z, w]

    return quaternions


def batch_matrices_to_quaternions_scipy_w_first(pose_matrices):
    """
    Use Scipy's Rotation library, return quaternions in [w, x, y, z] order

    Args:
        pose_matrices: numpy array of shape (n, 4, 4)

    Returns:
        numpy array of shape (n, 4), each row is a quaternion [w, x, y, z]
    """
    quaternions = batch_matrices_to_quaternions_scipy(pose_matrices)
    # Adjust order to [w, x, y, z]
    quaternions_w_first = np.zeros_like(quaternions)
    quaternions_w_first[:, 0] = quaternions[:, 3]  # w
    quaternions_w_first[:, 1:] = quaternions[:, :3]  # x, y, z
    return quaternions_w_first


class Light:
    def __init__(self, prim_path, stage, light_type, intensity, color, orientation, texture_file):
        self.prim_path = prim_path
        self.light_type = light_type
        self.stage = stage
        self.intensity = intensity
        self.color = color
        self.orientation = orientation
        base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/" + texture_file
        for file in os.listdir(base_folder):
            if file.endswith(".hdr"):
                self.texture_file = os.path.join(base_folder, file)

    def initialize(self):
        # selection between different light types
        if self.light_type == "Dome":
            light = UsdLux.DomeLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
            light.CreateTextureFileAttr().Set(Sdf.AssetPath(self.texture_file))
        elif self.light_type == "Sphere":
            light = UsdLux.SphereLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Disk":
            light = UsdLux.DiskLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Rect":
            light = UsdLux.RectLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)
        elif self.light_type == "Distant":
            light = UsdLux.DistantLight.Define(self.stage, Sdf.Path(self.prim_path))
            light.CreateIntensityAttr(self.intensity)
            light.CreateColorTemperatureAttr(self.color)

        light.CreateEnableColorTemperatureAttr().Set(True)
        lightPrim = XFormPrim(self.prim_path, orientation=self.orientation)

        return lightPrim


import omni.usd
from pxr import UsdGeom, Gf

# Isaac Sim utils: create prim helper
from omni.isaac.core.utils.prims import create_prim

def np_to_gf_matrix4d(T: np.ndarray) -> Gf.Matrix4d:
    """
    将机器人常见的齐次矩阵 T（通常满足 p' = R p + t，t 在最后一列）
    转成 USD/Gf 更匹配的矩阵约定。

    经验上最常见的坑：需要做一次转置，把“最后一列的平移”转到 USD 期望的位置。
    """
    T = np.asarray(T, dtype=np.float64)

    # 关键：约定转换（常见就是转置）
    # 如果 T 是标准 SE(3): [R t; 0 1]，那么转置后平移会跑到最后一行
    M = T.T

    # Gf.Matrix4d 的 16 参数构造按行填充（row-major）
    return Gf.Matrix4d(
        M[0, 0], M[0, 1], M[0, 2], M[0, 3],
        M[1, 0], M[1, 1], M[1, 2], M[1, 3],
        M[2, 0], M[2, 1], M[2, 2], M[2, 3],
        M[3, 0], M[3, 1], M[3, 2], M[3, 3],
    )


def set_local_matrix(prim, T_local_4x4: np.ndarray):
    """
    给 prim 设“局部变换矩阵”。
    使用 UsdGeom.Xformable 的 TransformOp（4x4 matrix op）。:contentReference[oaicite:3]{index=3}
    """
    xformable = UsdGeom.Xformable(prim)

    # 如果已有 transform op，就复用；否则新建
    ops = xformable.GetOrderedXformOps()
    transform_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = op
            break
    if transform_op is None:
        transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

    transform_op.Set(np_to_gf_matrix4d(T_local_4x4))


def set_display_color(prim, rgb: np.ndarray):
    """
    给 Gprim（Cube 属于 Gprim）设置 displayColor。
    displayColor 是最省事的上色方式，预览渲染通常直接显示。:contentReference[oaicite:4]{index=4}
    """
    rgb = np.asarray(rgb, dtype=np.float32).reshape(3,)
    gprim = UsdGeom.Gprim(prim)
    # displayColor 是 color3f[]，通常设置一个颜色即可
    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])


def remove_prim_if_exists(stage, prim_path: str):
    """如果 prim 存在则移除（便于重复运行脚本不堆叠）。"""
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def set_local_translate_scale(prim, t_xyz, s_xyz):
    """
    用明确的 xformOps 设置局部平移 + 缩放。
    关键点：
    - 清空 xformOp 顺序，避免历史 op 干扰
    - 使用自定义 opSuffix，避免与已有 xformOp:scale / xformOp:translate 撞名
    - precision 统一用 Double（避免 float/double 冲突）
    """
    xformable = UsdGeom.Xformable(prim)

    # 清空现有 op 顺序（不一定会删除属性，但会让后续 order 干净）
    xformable.ClearXformOpOrder()

    # 用自定义名字，避免撞上已有的 xformOp:translate / xformOp:scale
    # 统一用 Double，防止 precision 不一致异常
    t_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="translateLocal"
    )
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))

    s_op = xformable.AddXformOp(
        UsdGeom.XformOp.TypeScale,
        UsdGeom.XformOp.PrecisionDouble,
        opSuffix="scaleLocal"
    )
    s_op.Set(Gf.Vec3d(float(s_xyz[0]), float(s_xyz[1]), float(s_xyz[2])))


def spawn_axes_as_usd(
    T_batch: np.ndarray,
    root_prim_path: str = "/World",
    axes_container_name: str = "GraspsAxes",
    axis_len: float = 0.08,
    axis_thickness: float = 0.002,
    max_grasps: int | None = None,
):
    """
    生成 RGB 坐标轴(用 3 个 Cube),挂到另一个 group 下：
      /World/GraspsAxes/grasp_0000/x_axis
      /World/GraspsAxes/grasp_0000/y_axis
      /World/GraspsAxes/grasp_0000/z_axis

    与 grasps 的 id 完全一致(grasp_0000...)，便于对照。
    """
    N = int(T_batch.shape[0])
    if max_grasps is not None:
        N = min(N, int(max_grasps))
        T_batch = T_batch[:N]

    stage = omni.usd.get_context().get_stage()
    obj_prim = stage.GetPrimAtPath(root_prim_path)
    if not obj_prim or not obj_prim.IsValid():
        raise RuntimeError(f"找不到root prim: {root_prim_path}(请确认已加载到 Stage)")

    axes_root_path = f"{root_prim_path}/{axes_container_name}"
    remove_prim_if_exists(stage, axes_root_path)
    create_prim(axes_root_path, prim_type="Xform")

    # 三个轴的颜色
    col_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red
    col_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # green
    col_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # blue

    for i in range(N):
        # 用相同的 grasp id
        grasp_path = f"{axes_root_path}/grasp_{i:04d}"
        grasp_xf = create_prim(grasp_path, prim_type="Xform")

        # 坐标轴整体 pose 与 grasp pose 一致
        set_local_matrix(grasp_xf, T_batch[i])

        # x 轴：沿 +x，中心在 (axis_len/2,0,0)，尺寸 (axis_len, t, t)
        x_prim = create_prim(f"{grasp_path}/x_axis", prim_type="Cube")
        UsdGeom.Cube(x_prim).CreateSizeAttr().Set(1.0)
        set_local_translate_scale(
            x_prim,
            t_xyz=(axis_len / 2.0, 0.0, 0.0),
            s_xyz=(axis_len, axis_thickness, axis_thickness),
        )
        set_display_color(x_prim, col_x)

        # y 轴：沿 +y，中心在 (0,axis_len/2,0)，尺寸 (t, axis_len, t)
        y_prim = create_prim(f"{grasp_path}/y_axis", prim_type="Cube")
        UsdGeom.Cube(y_prim).CreateSizeAttr().Set(1.0)
        set_local_translate_scale(
            y_prim,
            t_xyz=(0.0, axis_len / 2.0, 0.0),
            s_xyz=(axis_thickness, axis_len, axis_thickness),
        )
        set_display_color(y_prim, col_y)

        # z 轴：沿 +z，中心在 (0,0,axis_len/2)，尺寸 (t, t, axis_len)
        z_prim = create_prim(f"{grasp_path}/z_axis", prim_type="Cube")
        UsdGeom.Cube(z_prim).CreateSizeAttr().Set(1.0)
        set_local_translate_scale(
            z_prim,
            t_xyz=(0.0, 0.0, axis_len / 2.0),
            s_xyz=(axis_thickness, axis_thickness, axis_len),
        )
        set_display_color(z_prim, col_z)

    print(f"[OK] 已生成{N} 个 target pose的可视化, 路径：{axes_root_path}")


def spawn_target_as_usd(
    target_position,
    target_rotation,
    root_prim_path: str = "/World",
    axes_container_name: str = "GraspsAxes",
    axis_len: float = 0.08,
    axis_thickness: float = 0.002,
):
    """
    target_position: (3,) xyz
    target_rotation: (4,) wxyz  (NOTE: scipy Rotation expects xyzw)

    生成一个 target 的坐标轴：实际复用 spawn_axes_as_usd，
    传入 batch=1 的 4x4 齐次变换矩阵。
    """
    # --- sanitize inputs ---
    p = np.asarray(target_position, dtype=np.float64).reshape(3,)
    q_wxyz = np.asarray(target_rotation, dtype=np.float64).reshape(4,)

    # scipy expects [x, y, z, w]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

    # optional: normalize quaternion to be safe
    n = np.linalg.norm(q_xyzw)
    if n == 0:
        raise ValueError("target_rotation quaternion has zero norm.")
    q_xyzw /= n

    # --- build transform ---
    rot = Rotation.from_quat(q_xyzw)         # xyzw
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = p

    T_batch = T[None, ...]  # shape (1, 4, 4)

    # --- reuse existing spawner ---
    return spawn_axes_as_usd(
        T_batch=T_batch,
        root_prim_path=root_prim_path,
        axes_container_name=axes_container_name,
        axis_len=axis_len,
        axis_thickness=axis_thickness,
        max_grasps=1,
    )

def clear_grasp_axes(
    root_prim_path: str = "/World",
    axes_container_name: str = "GraspsAxes",
):
    """
    删除 /World/GraspsAxes 这棵 prim 子树（如果存在）。

    设计目标：
    - 幂等：多次调用不报错
    - 精准：只删 GraspsAxes，不影响 World 下其他 prim
    """

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] USD Stage 未初始化，跳过 clear_grasp_axes")
        return

    axes_root_path = f"{root_prim_path}/{axes_container_name}"
    prim = stage.GetPrimAtPath(axes_root_path)

    if not prim or not prim.IsValid():
        # 不存在是完全正常的情况
        print("[WARN] clear prim invalid, 跳过删除 grasp axes")
        return

    # 使用 RemovePrim：USD 标准、安全
    stage.RemovePrim(axes_root_path)
    print(f"[OK] 已清除 grasp axes: {axes_root_path}")


# -----------------------------
# 核心：在 Isaac Sim 中生成 grasp 可视化
# -----------------------------
def spawn_grasps_as_usd(

    object_prim_path: str = "/World/Aligned",
    grasps_container_name: str = "Grasps",
    finger_len: float = 0.06,
    handle_len: float = 0.04,
    thickness: float = 0.006,
    seed: int = 0,
    max_grasps: int | None = None,
    direction_axis: str = "+x",  # fingers 伸出方向（原来 +x）
    grip_axis: str = "+y",       # width 张开方向（原来 +y）
):
    """
    方案A：在 Isaac Sim 的 Stage 里生成 grasp 可视化（实体夹爪，4个Cube）。

    参数：
    - direction_axis: fingers 沿该轴正方向伸出；handle 沿该轴负方向伸出
      例：'+x', '-x', '+y', '-z'
    - grip_axis: width 对称张开沿该轴
      例：'+y', '-y'（正负对张开本质等价，但用于统一你自己的定义）

    约束：
    - direction_axis 和 grip_axis 不能是同一根轴（否则几何会退化/重叠）
    """

    # -------- 轴解析工具（局部函数，避免污染全局）--------
    def _parse_axis(axis_str: str):
        s = axis_str.strip().lower()
        if len(s) == 1:
            sign = +1.0
            axis = s
        else:
            if s[0] == "+":
                sign = +1.0
                axis = s[1:]
            elif s[0] == "-":
                sign = -1.0
                axis = s[1:]
            else:
                # 允许用户直接输入 'x'/'y'/'z'
                sign = +1.0
                axis = s

        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis 只能是 x/y/z 或带符号的 +x/-y 等，收到：{axis_str}")

        idx = {"x": 0, "y": 1, "z": 2}[axis]
        v = np.zeros((3,), dtype=np.float64)
        v[idx] = sign
        return axis, idx, v

    def _orthogonal_axis_name(a: str, b: str):
        """给一个默认的“厚度方向”（第三轴），用于 thickness 放在平面外，视觉更稳定。"""
        axes = {"x", "y", "z"}
        rest = list(axes - {a, b})
        return rest[0]  # 剩下那根轴

    # -------- 解析轴 --------
    dir_axis_name, dir_idx, dir_vec = _parse_axis(direction_axis)  # (3,)
    grip_axis_name, grip_idx, grip_vec = _parse_axis(grip_axis)

    if dir_axis_name == grip_axis_name:
        raise ValueError(f"direction_axis({direction_axis}) 和 grip_axis({grip_axis}) 不能是同一根轴")

    # thickness 的“第三轴”（既不在 direction，也不在 grip）
    thick_axis_name = _orthogonal_axis_name(dir_axis_name, grip_axis_name)
    thick_idx = {"x": 0, "y": 1, "z": 2}[thick_axis_name]

    # 构造三个正交基（只用于决定 local translate/scale 的三个分量落在哪个轴上）
    # 注意：这里不改变 grasp_pose 的旋转定义，我们只是决定“夹爪几何在 grasp 局部系里沿哪根轴放置”
    # dir_vec / grip_vec 带符号，thick 方向无所谓符号，用 + 即可
    thick_vec = np.zeros((3,), dtype=np.float64)
    thick_vec[thick_idx] = 1.0

    # -------- 读 pkl --------
    T_batch, width = load_grasps_from_pkl(pkl_path)
    N = int(T_batch.shape[0])

    if max_grasps is not None:
        N = min(N, int(max_grasps))
        T_batch = T_batch[:N]
        width = width[:N]

    # -------- stage & object prim --------
    stage = omni.usd.get_context().get_stage()
    obj_prim = stage.GetPrimAtPath(object_prim_path)
    if not obj_prim or not obj_prim.IsValid():
        raise RuntimeError(f"找不到物体 prim：{object_prim_path}（请确认已加载到 Stage）")

    # -------- root prim --------
    grasps_root_path = f"{object_prim_path}/{grasps_container_name}"
    remove_prim_if_exists(stage, grasps_root_path)
    create_prim(grasps_root_path, prim_type="Xform")

    # -------- colors --------
    rng = np.random.RandomState(seed)
    colors = rng.uniform(low=0.2, high=0.9, size=(N, 3)).astype(np.float32)

    # -------- helpers：按轴构造 translate/scale --------
    def _make_t(offset_along_dir=0.0, offset_along_grip=0.0, offset_along_thick=0.0):
        """
        返回 (tx,ty,tz)：分别在 dir/grip/thick 三方向上的偏移，映射到 xyz 三个分量。
        """
        t = np.zeros((3,), dtype=np.float64)
        t += offset_along_dir * dir_vec
        t += offset_along_grip * grip_vec
        t += offset_along_thick * thick_vec
        return (float(t[0]), float(t[1]), float(t[2]))

    def _make_s(size_along_dir=0.0, size_along_grip=0.0, size_along_thick=0.0):
        """
        返回 (sx,sy,sz)：各向缩放尺寸映射到 xyz。
        """
        s = np.zeros((3,), dtype=np.float64)
        # 尺寸一定是正数
        s[dir_idx] = float(abs(size_along_dir))
        s[grip_idx] = float(abs(size_along_grip))
        s[thick_idx] = float(abs(size_along_thick))
        return (float(s[0]), float(s[1]), float(s[2]))

    # -------- build each grasp --------
    for i in range(N):
        grasp_path = f"{grasps_root_path}/grasp_{i:04d}"
        grasp_xf = create_prim(grasp_path, prim_type="Xform")

        # grasp 整体位姿（局部于 object）
        set_local_matrix(grasp_xf, T_batch[i])

        w = float(max(width[i], 1e-6))
        c = colors[i]

        # 1) handle：沿 -direction，长度 handle_len；中心在 (-handle_len/2)*direction
        handle_prim = create_prim(f"{grasp_path}/handle", prim_type="Cube")
        UsdGeom.Cube(handle_prim).CreateSizeAttr().Set(1.0)

        set_local_translate_scale(
            handle_prim,
            t_xyz=_make_t(offset_along_dir=-handle_len / 2.0),
            s_xyz=_make_s(size_along_dir=handle_len, size_along_grip=thickness, size_along_thick=thickness),
        )
        set_display_color(handle_prim, c)

        # 2) y_bar：沿 grip_axis，长度 w；中心在原点
        ybar_prim = create_prim(f"{grasp_path}/grip_bar", prim_type="Cube")
        UsdGeom.Cube(ybar_prim).CreateSizeAttr().Set(1.0)

        set_local_translate_scale(
            ybar_prim,
            t_xyz=_make_t(0.0, 0.0, 0.0),
            s_xyz=_make_s(size_along_dir=thickness, size_along_grip=w, size_along_thick=thickness),
        )
        set_display_color(ybar_prim, c)

        # 3) left finger：位于 grip 负半轴端点，沿 +direction 伸出
        fl_prim = create_prim(f"{grasp_path}/finger_L", prim_type="Cube")
        UsdGeom.Cube(fl_prim).CreateSizeAttr().Set(1.0)

        set_local_translate_scale(
            fl_prim,
            t_xyz=_make_t(offset_along_dir=+finger_len / 2.0, offset_along_grip=-w / 2.0),
            s_xyz=_make_s(size_along_dir=finger_len, size_along_grip=thickness, size_along_thick=thickness),
        )
        set_display_color(fl_prim, c)

        # 4) right finger：位于 grip 正半轴端点，沿 +direction 伸出
        fr_prim = create_prim(f"{grasp_path}/finger_R", prim_type="Cube")
        UsdGeom.Cube(fr_prim).CreateSizeAttr().Set(1.0)

        set_local_translate_scale(
            fr_prim,
            t_xyz=_make_t(offset_along_dir=+finger_len / 2.0, offset_along_grip=+w / 2.0),
            s_xyz=_make_s(size_along_dir=finger_len, size_along_grip=thickness, size_along_thick=thickness),
        )
        set_display_color(fr_prim, c)

    print(f"[OK] 已生成 grasps: {N} 个，路径：{grasps_root_path}")
    print(f"     direction_axis={direction_axis}, grip_axis={grip_axis}, thick_axis=+{thick_axis_name}")

