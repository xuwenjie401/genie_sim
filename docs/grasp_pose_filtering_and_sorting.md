# Grasp 位姿筛选与排序机制分析

日期：2026-04-05
范围：`PickStage.select_pose()` 及其调用的筛选/排序函数

## 1. 整体流程概览

`PickStage.select_pose()` 的完整 pipeline 可以分为三大阶段：**几何筛选 → IK 筛选 → IK 解排序**。

```
 canonical grasp poses (来自物体的 grasp 数据)
         │
         ▼
 ┌─────────────────────────┐
 │  Stage 1: 几何筛选       │  不涉及 IK，纯基于姿态几何
 │  (a) Z-percentile       │
 │  (b) gripper 坐标变换    │
 │  (c) set_grasp_vertical  │
 │  (d) flip_grasp          │
 │  (e) filter_grasp_pose   │
 │  (f) humanlike_filter    │
 │  (g) disable_upside_down │
 └─────────┬───────────────┘
           ▼
    random_downsample(300)
           │
           ▼
    apply grasp_offset
           │
           ▼
 ┌─────────────────────────┐
 │  Stage 2: IK 筛选        │  调用 IK 求解器做可行性判定
 │  (a) Simple IK           │
 │  (b) Next-stage IK 前瞻  │
 └─────────┬───────────────┘
           ▼
    random_downsample(100)
           │
           ▼
 ┌─────────────────────────┐
 │  Stage 3: IK 解排序       │  对通过所有筛选的候选进行排序
 │  (a) AvoidObs IK         │
 │  (b) 排序函数             │
 └─────────┬───────────────┘
           ▼
     返回排序后的 grasp pose 列表
```

以下分阶段详述。

---

## 2. Stage 1: 几何筛选

本阶段不调用 IK，只基于 grasp pose 的几何属性做过滤。每一步都是独立的 mask 操作，依次缩小候选集。

### 2.1 Z-percentile 过滤

代码位置：`grasp.py:67-74`

对 canonical 坐标系下 grasp pose 的 Y 分量（`grasp_poses_canonical[:, 1, 3]`，实际含义是抓取点在物体坐标系中的某个高度轴）做百分位过滤。

参数：
- `grasp_lower_percentile`（默认 0）
- `grasp_upper_percentile`（默认 100）

示例：当前 galbot 任务配置 `grasp_upper_percentile: 75`，表示只保留 Z 值在前 75% 的候选，删去最高的 25% 抓取点。

作用：粗略控制抓取高度范围，避免抓到物体最顶部或最底部不稳定的位置。

### 2.2 Gripper 坐标变换

代码位置：`grasp.py:76-78`

将 grasp 数据的旋转部分右乘 `robot.robot_gripper_2_grasp_gripper`，将 canonical grasp 坐标系转换为机器人夹爪坐标系。

对 galbot：此矩阵为单位阵（`omni_robot.py:73-75`），即 canonical 坐标系与 galbot 夹爪坐标系一致。

galbot 夹爪坐标系约定：
| 轴 | 含义 |
|---|---|
| X (column 0) | 夹爪前进方向 (approach direction) |
| Y (column 1) | 夹爪开合方向 |
| Z (column 2) | 夹爪手背朝向（"上"方向） |

### 2.3 set_grasp_vertical

代码位置：`grasp.py:79-86`

可选操作。将每个 grasp pose 的局部 Y 轴强制对齐到世界 Y 轴（右臂）或负 Y 轴（左臂），使夹爪开合方向水平。

通过计算当前 Y 轴到目标 Y 轴的旋转矩阵并左乘实现。

当前 galbot 任务：未配置，跳过。

### 2.4 flip_grasp

代码位置：`grasp.py:88-96`

可选操作。对每个 grasp pose 绕局部 Z 轴旋转 180°，生成"翻转抓取"候选，并与原候选合并。效果是将 grasp 数量翻倍。

当前 galbot 任务：`flip_grasp: false`，跳过。

### 2.5 世界坐标变换

代码位置：`grasp.py:102`

`grasp_poses = obj_pose @ grasp_poses_canonical`

将 canonical 坐标系下的 grasp pose 变换到世界坐标系。此后所有筛选和 IK 都在世界系下操作。

### 2.6 filter_grasp_pose（方向过滤）

代码位置：`grasp.py:104-108` → `common.py:203-306`

可选操作。通过 `extra_params.filter_grasp_pose` 配置，按夹爪某个轴与目标方向的夹角做过滤。

`filter_grasp_pose_by_gripper_up_direction()` 的逻辑：
1. 指定 `gripper_up_axis`（默认 `"y"`）：从 grasp pose 中提取哪个轴作为参考方向
2. 指定 `target_direction`（默认 `"x"`）：世界系中的目标方向
3. 指定 `threshold`（弧度）：保留夹角小于阈值的候选

语义：保留"夹爪某轴朝向与指定世界方向对齐"的候选。

当前 galbot 任务：未配置，跳过。

### 2.7 humanlike_filter

代码位置：`grasp.py:110-111` → `common.py:160-170`

可选操作。硬编码条件：

```python
mask = [pose[2, 1] > 0.0 and pose[0, 2] > 0 and pose[1, 2] > 0 for pose in grasp_poses]
```

三个条件的含义（在世界系下）：
- `pose[2, 1] > 0`：Y 轴的世界 Z 分量 > 0 → 夹爪开合方向有朝上分量
- `pose[0, 2] > 0`：Z 轴的世界 X 分量 > 0 → 手背方向有朝前分量
- `pose[1, 2] > 0`：Z 轴的世界 Y 分量 > 0 → 手背方向有朝右分量

这组条件是为特定机器人（非 galbot）设计的，对 galbot 不一定适用。

当前 galbot 任务：未配置，跳过。

### 2.8 disable_upside_down

代码位置：`grasp.py:113-129`

**对不同机器人有不同判定条件**：

| 机器人 | 条件 | 含义 |
|--------|------|------|
| omnipicker (左) | `grasp_poses[:, 2, 1] < 0` | Y 轴世界 Z 分量 < 0 |
| omnipicker (右) | `grasp_poses[:, 2, 1] > 0` | Y 轴世界 Z 分量 > 0 |
| agile | `grasp_poses[:, 2, 1] > 0` | Y 轴世界 Z 分量 > 0 |
| **galbot** | **`grasp_poses[:, 2, 2] > 0`** | **Z 轴世界 Z 分量 > 0 → 手背朝上** |
| 默认 | `grasp_poses[:, 2, 0] > 0` | X 轴世界 Z 分量 > 0 |

对 galbot 的语义：要求夹爪"手背"（Z 轴）的世界 Z 分量为正，即手背朝上、手心朝下。删去所有翻转的抓取姿态。

当前 galbot 任务：`disable_upside_down: true`，生效。

---

## 3. 降采样与 grasp_offset

### 3.1 第一次降采样

代码位置：`grasp.py:136-138`

`random_downsample(grasp_poses, 300)` — 如果经过几何筛选后候选数仍超过 300，随机采样 300 个。

目的：控制后续 IK 求解的计算量。

### 3.2 grasp_offset

代码位置：`grasp.py:141-149`

沿 grasp pose 的 **局部 Z 轴** 方向偏移抓取点位置：

```python
transport_vector = grasp_rotate @ [0, 0, 1, 0]   # 局部 Z 轴在世界系的方向
grasp_poses[:, :3, 3] += transport_vector * grasp_offset
```

> **注意**：对 galbot，夹爪的 approach direction 是 X 轴，但 grasp_offset 沿 Z 轴偏移（手背方向）。当 `grasp_offset = 0.0`（当前默认）时无影响，但若配置非零值，偏移方向与直觉的"接近方向"不一致。这是一个已知的轴定义不一致问题。

---

## 4. Stage 2: IK 筛选

### 4.1 Simple IK

代码位置：`grasp.py:152-157`

调用 `robot.solve_ik(grasp_poses, type="Simple")`。

- 后端：CuRobo
- 作用：快速判断每个 grasp pose 是否有可行的 IK 解
- 不考虑障碍物碰撞
- 每个目标 pose 返回 **1 个** IK 解（不返回多解）
- 返回 `ik_success` bool 数组，过滤掉无解的候选
- 此处的 `ik_info` 被丢弃（`_`），不用于后续

### 4.2 Next-stage IK 前瞻

代码位置：`grasp.py:231-413`

当后续 stage 是 place/pour 等放置类动作时，代码会评估：**如果用这个 grasp pose 抓住物体，后续放置时 IK 还能不能解出来？**

计算过程：
1. 遍历后续 stage 的 active_element（放置目标位姿集合）
2. 对每个 active_element，计算 `target_obj_poses`（物体在目标位姿的变换矩阵，通常有 N_align 个旋转采样）
3. `target_gripper_poses = target_obj_poses @ grasp_poses_canonical` — 在每个目标物体位姿下，夹爪应处于的位姿
4. 对 `target_gripper_poses` 做 Simple IK
5. 统计每个 grasp pose 在所有目标位姿上的 IK 成功次数 → `grasp_ik_score`

筛选条件：

```python
_mask = grasp_ik_score >= max(np.median(grasp_ik_score) / 2, 1)
```

即只保留 IK 成功次数 >= max(中位数/2, 1) 的 grasp pose。

作用：确保选出的 grasp pose 不仅抓得到，放置时也能做到。对 pick-place 类任务极为重要。

### 4.3 use_near_point 近点搜索（可选）

代码位置：`grasp.py:328-411`

当前瞻筛选导致候选清空时，如果配置了 `use_near_point: true`，会在目标放置位置周围做 XY 网格搜索（±0.06m，步长 0.02m），寻找附近可行的放置位置。

当前 galbot 任务：未配置。

---

## 5. Stage 3: IK 解排序

经过前两个阶段后，剩余候选已经是"几何合理 + IK 可行"的子集。本阶段目的是对这些候选**排出优劣顺序**。

### 5.1 第二次降采样

代码位置：`grasp.py:418`

`random_downsample(best_grasp_poses, 100)` — 进一步压缩到 100 个。

### 5.2 AvoidObs IK

代码位置：`grasp.py:424-436`

调用 `robot.solve_ik(best_grasp_poses, type="AvoidObs", output_link_pose=True)`。

- 后端：Isaac Sim collision-aware IK
- 比 Simple IK 更严格，考虑场景中的障碍物
- `output_link_pose=True`：额外返回所有 link 的世界坐标位姿

返回数据结构（`ik_info` dict）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `joint_positions` | `np.array (N, num_joints)` | 每个 IK 解的关节角度 |
| `joint_names` | `np.array (N, num_joints)` | 对应的关节名称 |
| `jacobian_score` | `np.array (N,)` | Yoshikawa 可操作度指标 |
| `link_poses` | `np.array (N,)` of dict | 每个 link 的 `[position_xyz, quaternion_wxyz]` |

### 5.3 Jacobian Score（可操作度指标）

计算位置：`client.py:419-430`

```python
pinocchio.forwardKinematics(model, data, joint_positions)
J = pinocchio.computeJointJacobian(model, data, joint_positions, joint_index)
manip = np.sqrt(np.linalg.det(np.dot(J, J.T)))
```

**数学定义**：Yoshikawa Manipulability Index

$$w = \sqrt{\det(J \cdot J^T)}$$

其中 $J$ 是在当前关节构型下的 6×n 几何 Jacobian 矩阵。

**物理含义**：
- 衡量机器人在当前构型下的末端运动能力
- 值越大 → 离奇异构型越远 → 末端在各方向都能灵活运动
- 值越小或为零 → 接近奇异构型 → 某些方向的运动能力丧失
- 直觉理解：Jacobian 的行列式是 J 列向量张成的超平行体的"体积"；体积越大，各方向覆盖越均匀

**galbot 特殊处理**：
- 对于 galbot 使用固定的 Pinocchio joint index = 27（对应 `right_arm_joint7`）
- 这意味着 **jacobian_score 始终相对于右臂末端计算**，即使当前 pick 使用左臂
- `client.py:425-427` 的 galbot 分支只写了 `27`（右臂），没有根据 `is_right` 参数区分左右臂
- 这是一个潜在 bug：左臂 pick 时，jacobian_score 反映的是右臂的可操作度，没有实际意义

### 5.4 当前生效的排序函数

代码位置：`grasp.py:482-491` → `common.py:80-144`

**`sorted_by_joint_pos_dist_and_grasp_pose()`** — 尽管函数名暗示结合了关节距离和抓取方向，但实际实现只用了关节距离。

计算过程：

```python
# 1. 取当前关节位置
cur_joint_positions = robot.get_current_joint_positions(arm)

# 2. 对每个 IK 解，计算关节空间距离（标准化后）
joint_pos_dist = ||target_joint_positions - cur_joint_positions||_2
joint_pos_dist = (joint_pos_dist - mean) / std

# 3. Jacobian score 归一化到 [0, 1]（但没用到！）
ik_jacobian_score = normalize_0_1(ik_jacobian_score)

# 4. cost = 只有关节距离
cost = joint_pos_dist         # ← jacobian_score 被忽略了
idx_sorted = argsort(cost)    # 升序：关节运动最少的排最前
```

**结果**：当前 galbot 的 grasp pose 排序 **纯粹按"哪个 IK 解离当前关节位姿最近"排序**，不考虑可操作度、抓取方向、肘部位置等任何其他因素。

### 5.5 被禁用的 humanlike 排序

代码位置：`grasp.py:467-480`（`if False:` 分支），`sort_pose.py:193-257`

`sorted_by_position_humanlike()` 是一个更复杂的排序函数，曾为 G2 机器人设计，当前被禁用。

它的排序 cost 包含三项：

| 项 | 权重 | 含义 |
|---|---|---|
| `elbow_y` × `weight_elbow_out` | 1.0 | 肘部外展程度（左臂越负越好，右臂越正越好） |
| `elbow_x` × `weight_elbow_back` | 0.5 | 肘部前后位置（越靠后越好，即 x 越小越好） |
| `hand_z_score` | 0.1 | 手部 Z 轴方向评分（基于分段线性评分器） |

**注意：此函数不检查肘部高度（elbow_z）。**

#### hand_z_score 的评分体系

手部 Z 轴方向被分解为 azimuth（水平角）和 elevation（仰角），分别通过 `PiecewiseScorer` 评分。

**Azimuth 评分（右臂）**：
```
最优区间: 70°~100° (cost=0) → Z 轴朝左，夹爪手背面向左侧
从 100° 到 270°: cost 线性增加 (weight=1.0) → 手背逐渐转向右侧，越来越差
从 270° 到 360°: cost 线性降低 (weight=-1.5) → 手背转回前方
从 0° 到 70°: cost 线性降低 (weight=-0.5) → 手背朝前偏左
```

**Azimuth 评分（左臂）** — 与右臂镜像对称：
```
最优区间: 260°~290° (cost=0) → Z 轴朝右，夹爪手背面向右侧
```

**Elevation 评分（左右臂相同）**：
```
最优区间: -45°~5° (cost=0) → 手背大致水平或略微朝上
-90°~-45°: cost 增加 (weight=-0.5) → 手背朝上太多
5°~90°: cost 增加 (weight=0.5) → 手背朝下
```

语义总结：偏好**手背朝侧面、大致水平**的抓取姿态，即类似人类自然伸手拿东西的方式。

---

## 6. IK 解的关节构型分析

### 6.1 当前现状

当前 pipeline 中 **没有** 对 IK 解的关节构型（如肘部高度）做任何筛选或惩罚。

IK 求解器（CuRobo / Isaac Sim）对每个目标 pose 只返回 **1 个** IK 解（不提供多解选择），而该解的关节构型取决于求解器内部的初始值和优化路径，外部无法控制。

因此 pipeline 的"筛选"实际上是：
- 生成大量不同的 **grasp pose**（末端位姿不同）
- 每个 grasp pose 只有 **1 个** IK 解
- 排序只看**关节距离**，不看构型质量

这意味着如果某个 grasp pose 恰好解出肘部抬很高的构型，只要它离当前关节位姿近，就会排在前面。

### 6.2 link_poses 中可用的数据

AvoidObs IK 返回的 `ik_link_poses` 包含所有 link 的世界坐标位姿，可以直接用于关节构型分析。

galbot 手臂 link 结构（URDF 运动链）：

```
torso_{left,right}_arm_mount_link (固定)
  └─ {left,right}_arm_base_link (固定)
      └─ {left,right}_arm_link1 (joint1: 肩部旋转)
          └─ {left,right}_arm_link2 (joint2: 肩部俯仰)
              └─ {left,right}_arm_link3 (joint3: 上臂旋转)
                  └─ {left,right}_arm_link4 (joint4: 肘部弯曲) ← "肘关节"
                      └─ {left,right}_arm_link5 (joint5: 前臂旋转)
                          └─ {left,right}_arm_link6 (joint6: 腕部俯仰)
                              └─ {left,right}_arm_link7 (joint7: 腕部旋转)
```

> **注意**：`grasp.py:471` 中被禁用的代码使用 `arm_r_link4` / `arm_l_link4` 作为 elbow link 名。但 galbot URDF 中的实际名称是 `right_arm_link4` / `left_arm_link4`。如果后续启用 humanlike 排序或新增肘部筛选，需要使用正确的名称。

### 6.3 可用于肘部筛选的数据提取

在 AvoidObs IK 之后，可以从 `ik_link_poses` 提取肘部和肩部世界坐标：

```python
# 伪代码，说明数据如何获取
elbow_name = "left_arm_link4" if arm == "left" else "right_arm_link4"
shoulder_name = "left_arm_link2" if arm == "left" else "right_arm_link2"

for i in range(len(ik_link_poses)):
    elbow_pos = ik_link_poses[i][elbow_name][0]    # np.array [x, y, z]
    shoulder_pos = ik_link_poses[i][shoulder_name][0]
    elbow_z = elbow_pos[2]
    shoulder_z = shoulder_pos[2]
```

---

## 7. 现有问题汇总

| # | 问题 | 影响 | 位置 |
|---|------|------|------|
| 1 | `sorted_by_joint_pos_dist_and_grasp_pose` 函数名误导 | 函数名暗示结合了 grasp pose 方向信息，实际只用关节距离 | `common.py:80-144` |
| 2 | jacobian_score 计算但未使用 | `sorted_by_joint_pos_dist_and_grasp_pose` 对 jacobian_score 做了归一化但没加入 cost | `common.py:141` |
| 3 | galbot jacobian_score 的左右臂 bug | Pinocchio joint index 固定为 27（右臂），左臂 pick 时计算的是右臂的可操作度 | `client.py:425-427` |
| 4 | 无肘部高度筛选 | IK 解的肘部抬高程度完全不受控 | `grasp.py:482-491` |
| 5 | humanlike 排序被禁用且 link 名不匹配 galbot | `if False:` 分支中使用 `arm_r_link4`，galbot URDF 中为 `right_arm_link4` | `grasp.py:467-471` |
| 6 | grasp_offset 沿 Z 轴而非 approach direction | 对 galbot（approach=X）偏移方向不正确（当 offset=0 时无影响） | `grasp.py:141-147` |
| 7 | `sorted_by_joint_pos_dist` 与 `sorted_by_joint_pos_dist_and_grasp_pose` 行为差异 | 前者 `cost = joint_dist - jacobian_score`（用了 jacobian），后者 `cost = joint_dist`（没用） | `common.py:73` vs `common.py:141` |
