import pickle
import numpy as np
import open3d as o3d


def load_grasps(pkl_path: str):
    """读取 pkl，并返回 (T_batch, width)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"pkl 顶层不是 dict，而是 {type(data)}")

    if "grasp_pose" not in data:
        raise KeyError("pkl 缺少 key: 'grasp_pose'")
    if "width" not in data:
        raise KeyError("pkl 缺少 key: 'width'")

    T = np.asarray(data["grasp_pose"], dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"'grasp_pose' 形状应为 (N,4,4)，实际为 {T.shape}")

    width = np.asarray(data["width"], dtype=np.float64)
    if width.ndim != 1 or width.shape[0] != T.shape[0]:
        raise ValueError(f"'width' 形状应为 (N,)，且 N={T.shape[0]}；实际为 {width.shape}")

    return T, width


def transform_mesh(mesh: o3d.geometry.TriangleMesh, T: np.ndarray) -> o3d.geometry.TriangleMesh:
    """对 mesh 应用 4x4 变换（原地修改后返回，便于链式调用）。"""
    mesh.transform(T)
    return mesh


def create_centered_box(size_x: float, size_y: float, size_z: float) -> o3d.geometry.TriangleMesh:
    """
    创建一个以原点为中心、轴对齐的 box mesh。
    Open3D 的 create_box 默认从 (0,0,0) 到 (x,y,z)，所以要平移到中心。
    """
    box = o3d.geometry.TriangleMesh.create_box(width=size_x, height=size_y, depth=size_z)
    box.translate(np.array([-size_x / 2.0, -size_y / 2.0, -size_z / 2.0], dtype=np.float64))
    return box


def build_gripper_mesh_for_one_grasp(
    T: np.ndarray,
    w: float,
    finger_len: float,
    handle_len: float,
    bar_thickness: float,
    color_rgb: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """
    为单个 grasp 构建“实体夹爪（box 组合）”，然后用 T 放到世界坐标系。

    你的语义要求（局部系）：
    - Y 轴横线：开口宽度（[-w/2, +w/2]）
    - X 负方向：夹爪把手（handle）
    - Y 横线两端：沿 X 正方向两条线表示 gripper（这里用 box finger）

    这里用 4 个 box：
    1) handle：沿 X 负方向，长度 handle_len
    2) y_bar：沿 Y 方向，长度 w
    3) left finger：从 (0, -w/2, 0) 沿 +X，长度 finger_len
    4) right finger：从 (0, +w/2, 0) 沿 +X，长度 finger_len
    """
    # 为避免宽度极小导致几何退化，这里做个下限（你也可以去掉）
    w = float(max(w, 1e-6))

    # 1) handle：尺寸 (handle_len, t, t)，中心在 (-handle_len/2, 0, 0)
    handle = create_centered_box(handle_len, bar_thickness, bar_thickness)
    handle.translate(np.array([-handle_len / 2.0, 0.0, 0.0], dtype=np.float64))

    # 2) y_bar：尺寸 (t, w, t)，中心在 (0, 0, 0)
    y_bar = create_centered_box(bar_thickness, w, bar_thickness)
    # y_bar 中心已在 (0,0,0)，无需再平移

    # 3) left finger：尺寸 (finger_len, t, t)，中心在 (finger_len/2, -w/2, 0)
    left_finger = create_centered_box(finger_len, bar_thickness, bar_thickness)
    left_finger.translate(np.array([finger_len / 2.0, -w / 2.0, 0.0], dtype=np.float64))

    # 4) right finger：尺寸 (finger_len, t, t)，中心在 (finger_len/2, +w/2, 0)
    right_finger = create_centered_box(finger_len, bar_thickness, bar_thickness)
    right_finger.translate(np.array([finger_len / 2.0, +w / 2.0, 0.0], dtype=np.float64))

    # 合并为一个 mesh（同一夹爪统一颜色更干净）
    gripper = handle + y_bar + left_finger + right_finger
    gripper.paint_uniform_color(color_rgb.tolist())
    gripper.compute_vertex_normals()

    # 放到世界坐标系
    transform_mesh(gripper, T)
    return gripper


def build_mode1_geometries(T_batch: np.ndarray, width: np.ndarray, frame_size: float):
    """
    模式1：RGB坐标轴 + 每个 grasp 原点处沿 y 轴对称 width 线段（绿色）。
    """
    geoms = []
    N = T_batch.shape[0]

    # 坐标轴
    for i in range(N):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(T_batch[i])
        geoms.append(frame)

    # width 线段（LineSet）
    # 每个 grasp 两端点： (0,-w/2,0) 与 (0,+w/2,0) ，再用 T 变换
    points = np.zeros((2 * N, 3), dtype=np.float64)
    lines = np.zeros((N, 2), dtype=np.int32)
    colors = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (N, 1))  # 绿色

    for i in range(N):
        w = float(width[i])
        p_local = np.array([[0.0, -0.5 * w, 0.0],
                            [0.0, +0.5 * w, 0.0]], dtype=np.float64)

        # 变换点
        M = p_local.shape[0]
        p_h = np.hstack([p_local, np.ones((M, 1), dtype=np.float64)])
        p_w = (T_batch[i] @ p_h.T).T[:, :3]

        points[2 * i: 2 * i + 2] = p_w
        lines[i] = [2 * i, 2 * i + 1]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    geoms.append(ls)
    return geoms


def main():
    pkl_path = input("请输入 pkl 文件路径：").strip().strip('"').strip("'")
    pkl_path = "/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_beverage_bottle_001/grasp_pose/grasp_pose.pkl"
    if not pkl_path:
        print("路径为空，退出。")
        return

    mode = input("选择模式：1=RGB坐标轴+width线段，2=实体夹爪box（默认2）：").strip()
    mode = mode if mode in ("1", "2") else "2"

    print("\n[1] 读取 pkl ...")
    T_batch, width = load_grasps(pkl_path)
    N = T_batch.shape[0]
    print(f"[2] grasp 数量 N={N}, width已知沿y轴")

    geoms = []

    if mode == "1":
        frame_size_s = input("坐标轴大小 frame_size（回车默认 0.05）：").strip()
        frame_size = float(frame_size_s) if frame_size_s else 0.05
        geoms = build_mode1_geometries(T_batch, width, frame_size)
        win_name = f"Mode1: RGB axes + width(y) segments (N={N})"

    else:
        # ===== 模式2：实体夹爪 box（粗、清晰、每个夹爪不同颜色）=====
        finger_len_s = input("finger_len（沿 +X finger 长度，回车默认）：").strip()
        handle_len_s = input("handle_len（沿 -X handle 长度，回车默认）：").strip()
        thickness_s  = input("bar_thickness（线条粗细/box厚度，回车默认）：").strip()
        seed_s       = input("颜色随机种子 seed（回车默认 0）：").strip()

        finger_len = float(finger_len_s) if finger_len_s else 0.01
        handle_len = float(handle_len_s) if handle_len_s else 0.001
        thickness  = float(thickness_s)  if thickness_s  else 0.001
        seed       = int(seed_s)         if seed_s       else 0

        rng = np.random.RandomState(seed)
        # 每个夹爪一个颜色：限制到 [0.2, 0.9] 避免太暗或太亮
        grasp_colors = rng.uniform(low=0.2, high=0.9, size=(N, 3))

        # N = min(N, 100)
        for i in range(N):
            g = build_gripper_mesh_for_one_grasp(
                T=T_batch[i],
                w=float(width[i]),
                finger_len=finger_len,
                handle_len=handle_len,
                bar_thickness=thickness,
                color_rgb=grasp_colors[i],
            )
            geoms.append(g)

        win_name = f"Mode2: Thick box-grippers (N={N})"

    print("[3] 打开窗口显示 ...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=win_name,
        width=1200,
        height=900
    )


if __name__ == "__main__":
    main()



