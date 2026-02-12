import pickle
import numpy as np

import isaacsim

# Standard Library
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, default="/World/Aligned")
parser.add_argument("--finger_len", type=float, default=0.06)
parser.add_argument("--handle_len", type=float, default=0.04)
parser.add_argument("--thickness", type=float, default=0.006)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_grasps", type=int, default=-1)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "width": 1920,   # ✅ 用 int
        "height": 1080,  # ✅ 用 int
        "headless": False
    }
)

import omni.usd
from pxr import UsdGeom, Gf

# Isaac Sim utils: create prim helper
from omni.isaac.core.utils.prims import create_prim


# PKL_PATH = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_beverage_bottle_001/grasp_pose/grasp_pose.pkl"
# USD_PATH = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/beverage_bottle/benchmark_beverage_bottle_001/Aligned.usda"

PKL_PATH = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_beverage_bottle_008/grasp_pose/grasp_pose.pkl"
USD_PATH = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/beverage_bottle/benchmark_beverage_bottle_008/Aligned.usda"

# -----------------------------
# 基础工具函数
# -----------------------------
def load_grasps_from_pkl(pkl_path: str):
    """读取 pkl: 期望 dict {'grasp_pose': (N,4,4), 'width': (N,)}"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"pkl 顶层不是 dict，而是 {type(data)}")

    if "grasp_pose" not in data or "width" not in data:
        raise KeyError("pkl 需要包含 'grasp_pose' 和 'width'")

    T = np.asarray(data["grasp_pose"], dtype=np.float64)
    w = np.asarray(data["width"], dtype=np.float64)

    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"grasp_pose 应为 (N,4,4)，实际 {T.shape}")
    if w.ndim != 1 or w.shape[0] != T.shape[0]:
        raise ValueError(f"width 应为 (N,)，且 N={T.shape[0]}，实际 {w.shape}")

    return T, w


def rotz(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0,0.0, 1.0]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


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


# def np_to_gf_matrix4d(T: np.ndarray) -> Gf.Matrix4d:
#     """
#     保持旋转 R 不动，只调整平移的存放位置：
#     - 输入：t 在最后一列（机器人常见）
#     - 输出：t 放到最后一行（常见于行向量约定/某些系统的矩阵解释）
#     """
#     T = np.asarray(T, dtype=np.float64)

#     R = T[:3, :3]
#     t = T[:3, 3]

#     M = np.eye(4, dtype=np.float64)
#     M[:3, :3] = R

#     # 只搬平移：放到最后一行前三个元素
#     M[3, 0:3] = t
#     M[0:3, 3] = 0.0  # 清掉最后一列平移（避免混乱）

#     return Gf.Matrix4d(
#         M[0, 0], M[0, 1], M[0, 2], M[0, 3],
#         M[1, 0], M[1, 1], M[1, 2], M[1, 3],
#         M[2, 0], M[2, 1], M[2, 2], M[2, 3],
#         M[3, 0], M[3, 1], M[3, 2], M[3, 3],
#     )


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


def make_scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """构造 4x4 缩放矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = sx
    T[1, 1] = sy
    T[2, 2] = sz
    return T


def make_translate_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """构造 4x4 平移矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def compose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """4x4 矩阵相乘：先应用 T_b 再应用 T_a（标准列向量形式时是 T_a @ T_b）"""
    return (T_a @ T_b).astype(np.float64)


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


# -----------------------------
# 核心：在 Isaac Sim 中生成 grasp 可视化
# -----------------------------
def spawn_grasps_as_usd(
    pkl_path: str,
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


def spawn_axes_as_usd(
    T_batch: np.ndarray,
    object_prim_path: str = "/World/Aligned",
    axes_container_name: str = "GraspsAxes",
    axis_len: float = 0.08,
    axis_thickness: float = 0.002,
    max_grasps: int | None = None,
):
    """
    生成 RGB 坐标轴（用 3 个 Cube），挂到另一个 group 下：
      /World/Aligned/GraspsAxes/grasp_0000/x_axis
      /World/Aligned/GraspsAxes/grasp_0000/y_axis
      /World/Aligned/GraspsAxes/grasp_0000/z_axis

    与 grasps 的 id 完全一致（grasp_0000...），便于对照。
    """
    N = int(T_batch.shape[0])
    if max_grasps is not None:
        N = min(N, int(max_grasps))
        T_batch = T_batch[:N]

    stage = omni.usd.get_context().get_stage()
    obj_prim = stage.GetPrimAtPath(object_prim_path)
    if not obj_prim or not obj_prim.IsValid():
        raise RuntimeError(f"找不到物体 prim：{object_prim_path}（请确认已加载到 Stage）")

    axes_root_path = f"{object_prim_path}/{axes_container_name}"
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

    print(f"[OK] 已生成坐标轴: {N} 个，路径：{axes_root_path}")


def load_object_usd(
    usd_path: str,
    prim_path: str = "/World/Aligned",
):
    """
    将一个 USD 文件加载到当前 Stage，并挂载到 prim_path。
    - 如果 prim_path 已存在，会先删除再加载（避免叠加）
    """
    stage = omni.usd.get_context().get_stage()

    # 如果已存在，先移除
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)

    # 创建 Xform prim
    obj_prim = create_prim(prim_path, prim_type="Xform")

    # 通过 reference 的方式加载 USD
    obj_prim.GetReferences().AddReference(usd_path)

    print(f"[OK] 已加载 USD: {usd_path}")
    print(f"     挂载到 prim: {prim_path}")


def main():
    pkl_path = PKL_PATH
    usd_path = USD_PATH

    for _ in range(30):
        simulation_app.update()

    # 1) 先加载物体 USD
    load_object_usd(
        usd_path=usd_path,
        prim_path="/World/Aligned",
    )

    # ⚠️ 很重要：推进几帧，让 reference 真正被 stage compose + viewport 刷新
    for _ in range(10):
        simulation_app.update()

    # 2) 读参数（给一个更“可见”的默认值）
    # finger_len_s = input("finger_len (回车默认 0.06): ").strip()
    # handle_len_s = input("handle_len (回车默认 0.04): ").strip()
    # thickness_s  = input("thickness  (回车默认 0.006): ").strip()
    # seed_s       = input("颜色 seed   (回车默认 0): ").strip()
    # max_s        = input("最多显示多少个 grasp（回车=全部）: ").strip()
    finger_len_s = 0.04
    handle_len_s = 0.08
    thickness_s  = 0.003
    seed_s       = 0
    max_s        = None

    finger_len = float(finger_len_s) if finger_len_s else 0.04
    handle_len = float(handle_len_s) if handle_len_s else 0.08
    thickness  = float(thickness_s)  if thickness_s  else 0.003
    seed       = int(seed_s)         if seed_s       else 0
    max_grasps = int(max_s)          if max_s        else None

    # 3) 生成 grasps
    spawn_grasps_as_usd(
        pkl_path=pkl_path,
        object_prim_path="/World/Aligned",
        grasps_container_name="Grasps",
        finger_len=finger_len,
        handle_len=handle_len,
        thickness=thickness,
        seed=seed,
        max_grasps=max_grasps,
    )

    # ⚠️ 再推进几帧，确保新 prim 出现在视窗
    for _ in range(10):
        simulation_app.update()

    # 额外生成坐标轴组（与 grasps 同 id 对照）
    # 注意：这里要用同一批 grasp_pose，所以直接从 pkl 再读一次（简单可靠）
    T_batch, _ = load_grasps_from_pkl(pkl_path)

    # T = rotz(90.0)
    # T[:3,3] = np.array([0.3, 0.0, 0.0])
    # T_batch = np.array([T], dtype=np.float64)

    spawn_axes_as_usd(
        T_batch=T_batch,
        object_prim_path="/World/Aligned",
        axes_container_name="GraspsAxes",   # 另一个 group
        axis_len=0.06,                      # 轴长度（按你物体尺度调）
        axis_thickness=0.002,               # 轴粗细
        max_grasps=max_grasps,              # 和 grasps 保持一致的数量
    )

    # ⚠️ 再推进几帧，确保新 prim 出现在视窗
    for _ in range(10):
        simulation_app.update()

    # 4) 简单验证：检查一个 cube prim 是否真的存在于 stage
    stage = omni.usd.get_context().get_stage()
    test_path = "/World/Aligned/Grasps/grasp_0000/handle"
    test_prim = stage.GetPrimAtPath(test_path)
    print(f"[CHECK] {test_path} valid? -> {bool(test_prim and test_prim.IsValid())}")

    print("\n窗口已打开。关闭窗口或 Ctrl+C 退出。\n")

    # 5) 关键：保持渲染循环，不然你“只看到物体/看不到更新”
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()

