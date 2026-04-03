# Place动作位姿生成与选择流程详解

## 概述

Place动作负责将抓取的物体放置到目标位置。本文档详细解释`place.py`中位姿生成和选择的完整流程。

## 核心流程

### 1. 入口：`select_pose()`

**输入参数：**
- `objects`: 场景中的所有物体
- `robot`: 机器人实例

**关键变量初始化：**
```python
object_pose = objects[self.active_obj_id].obj_pose  # 当前抓取物体的位姿
ee_pose = robot.get_ee_pose(ee_type="gripper", id=arm)  # 末端执行器位姿
gripper2obj = np.linalg.inv(object_pose) @ ee_pose  # 夹爪到物体的相对变换
```

---

## 2. 位姿生成阶段

### 2.1 获取交互元素

从任务配置中获取：
- `active_element`: 主动物体的交互元素（被抓取的物体）
- `passive_element`: 被动物体的交互元素（目标容器，如收纳盒）

### 2.2 生成对齐位姿

```python
# 更新物体的对齐信息
active_obj.update_aligned_info(active_element)
passive_obj.update_aligned_info(passive_element)

# 计算对齐采样数量（最小公倍数）
N_align = np.lcm(passive_obj.angle_sample_num, active_obj.angle_sample_num)

# 生成对齐位姿
if "fix_pose" in self.passive_obj_id:
    target_obj_poses = get_aligned_fix_pose(active_obj, passive_obj, N=N_align)
else:
    target_obj_poses = get_aligned_pose(active_obj, passive_obj, N=N_align)
```

**关键点：** `get_aligned_pose()`根据物体的标注信息（annotation）生成多个可能的放置位姿。

### 2.3 转换为夹爪位姿

```python
target_gripper_poses = target_obj_poses @ gripper2obj[np.newaxis, ...]
```

将物体位姿转换为对应的夹爪位姿，保持抓取时的相对关系。

---

## 3. 位姿过滤阶段

### 3.1 姿态过滤（防止倒置）

```python
if disable_upside_down:
    if "omnipicker" in robot.robot_cfg:
        if arm == "left":
            upright_mask = target_gripper_poses[:, 2, 1] < 0.0
        else:
            upright_mask = target_gripper_poses[:, 2, 1] > 0.0
    else:
        upright_mask = target_gripper_poses[:, 2, 0] > 0.0
    target_gripper_poses = target_gripper_poses[upright_mask]
```

过滤掉会导致物体倒置的位姿。

### 3.2 中心距离排序（第一次）

```python
center_idx_sorted = sort_place_pose_indices_by_center(target_obj_poses, anchor_pose)
target_gripper_poses = target_gripper_poses[center_idx_sorted]
if target_gripper_poses.shape[0] > center_sort_num:
    target_gripper_poses = target_gripper_poses[:center_sort_num]
```

**`sort_place_pose_indices_by_center()`函数逻辑：**
- 计算每个候选位姿到目标容器中心的距离
- 优先选择靠近中心的位姿
- 限制候选数量（默认100个）

---

## 4. IK求解阶段

### 4.1 简单IK预筛选

```python
ik_success, _ = robot.solve_ik(
    target_gripper_poses,
    ee_type="gripper",
    type="Simple",  # 不考虑碰撞
    arm=arm,
)
target_gripper_poses = target_gripper_poses[ik_success]
```

快速过滤掉运动学上不可达的位姿。

### 4.2 避障IK求解

```python
ik_success, ik_info = robot.solve_ik(
    target_gripper_poses,
    ee_type="gripper",
    type="AvoidObs",  # 考虑碰撞避障
    arm=arm,
    output_link_pose=True,
)

target_gripper_poses_pass_ik = target_gripper_poses[ik_success]
ik_joint_positions = ik_info["joint_positions"][ik_success]
ik_joint_names = ik_info["joint_names"][ik_success]
ik_jacobian_score = ik_info["jacobian_score"][ik_success]
ik_link_poses = ik_info["link_poses"][ik_success]
```

使用CuRobo进行完整的IK求解，考虑碰撞避障。

### 4.3 近点搜索（可选）

如果所有位姿都无法通过IK，启用`use_near_point`选项：

```python
for offset_axis in ["x", "y", "z"]:
    (target_gripper_poses_pass_ik, ...) = find_near_point_grasp_pose(
        robot, arm, target_gripper_poses,
        offset_range=np.linspace(0, 0.2, 5),
        offset_axis=offset_axis,
    )
```

在原始位姿附近搜索可行的替代位姿。

---

## 5. Pre-Place位姿处理（可选）

如果启用`use_pre_place`，会生成预放置位姿：

### 5.1 计算预放置位姿

```python
normal_direction = np.array(self.passive_element[0]["direction"])
pre_insert_pose_canonical = target_obj_pose_canonical.copy()
pre_insert_pose_canonical[:, :3, 3] += -normal_direction * self.pre_insert_offset
```

在放置点的法向方向上偏移一定距离（默认0.1m）。

### 5.2 添加噪声（可选）

```python
if "pre_pose_noise" in self.extra_params:
    position_noise = self.extra_params["pre_pose_noise"].get("position_noise", 0)
    rotation_noise = self.extra_params["pre_pose_noise"].get("rotation_noise", 0)
    pre_insert_pose_canonical = add_noise(...)
```

为预放置位姿添加随机噪声，增加数据多样性。

### 5.3 验证预放置位姿

```python
ik_success_pre, ik_info_pre = robot.solve_ik(
    pre_insert_gripper_pose,
    ee_type="gripper",
    type="AvoidObs",
    arm=arm,
)
```

只保留预放置位姿也能通过IK的候选。

---

## 6. 位姿排序阶段

### 6.1 人形化排序（针对G2机器人）

```python
if "G2" in robot.robot_cfg:
    idx_sorted = sorted_by_position_humanlike(
        joint_positions=ik_joint_positions,
        joint_names=ik_joint_names,
        link_poses=ik_link_poses,
        is_right=arm == "right",
        elbow_name=elbow_name,
        hand_name=hand_name,
    )
```

根据人类运动习惯排序（肘部和手部位置）。

### 6.2 关节距离排序（其他机器人）

```python
else:
    idx_sorted = sorted_by_joint_pos_dist(
        robot, arm, ik_joint_positions, ik_joint_names, ik_jacobian_score
    )
```

根据当前关节位置的距离和雅可比得分排序。

### 6.3 中心距离二次排序

```python
center_weight = self.extra_params.get("place_center_weight", 0.5)
idx_sorted = sort_place_pose_indices_by_center(
    target_obj_pose_pass_ik,
    anchor_pose,
    base_idx_sorted=idx_sorted,
    center_weight=center_weight,
)
```

**混合排序策略：**
```
final_score = base_rank + center_weight * center_score
```

- `base_rank`: 基于关节距离/人形化的排序
- `center_score`: 基于到容器中心距离的排序
- `center_weight`: 权重系数（默认0.5）

---

## 7. 结果输出

### 7.1 转换为规范坐标系

```python
target_obj_pose_sorted = target_obj_pose_pass_ik[idx_sorted]
target_obj_pose_canonical_sorted = np.linalg.inv(anchor_pose)[np.newaxis, ...] @ target_obj_pose_sorted
```

将位姿转换到目标容器的局部坐标系。

### 7.2 构建结果

```python
result = []
for i in range(len(target_obj_pose_canonical_sorted)):
    tmp_result = {}
    tmp_result["grasp_pose"] = target_obj_pose_canonical_sorted[i]
    if pre_insert_pose_canonical is not None:
        tmp_result["pre_insert_pose"] = pre_insert_pose_canonical[i]
    result.append(tmp_result)
```

返回排序后的候选位姿列表，每个包含：
- `grasp_pose`: 最终放置位姿
- `pre_insert_pose`: 预放置位姿（如果启用）

---

## 8. 动作序列生成：`generate_action_sequence()`

根据选定的位姿生成具体的动作序列：

### 8.1 基本放置动作

```python
palce_transform_up = np.eye(4)
palce_transform_up[:3, 3] = self.place_transform_up  # [0, 0, 0.05]
action_sequence.add_action(Action(target_pose_canonical, None, palce_transform_up, "AvoidObs"))
```

先移动到放置点上方5cm处。

### 8.2 后处理动作（可选）

```python
if post_place_action is not None:
    for post_action in post_place_action:
        # 1. 可选的夹爪动作
        post_place_gripper_cmd = post_action.get("gripper_cmd", None)
        
        # 2. 沿指定方向移动
        post_place_distance = post_action.get("distance", 0.02)
        post_place_direction = np.array(post_action.get("direction", [0, 0, 1]))
        target_pose_canonical[:3, 3] += post_place_direction * post_place_distance
```

### 8.3 释放物体

```python
gripper_cmd = self.extra_params.get("gripper_state", "open")
action_sequence.add_action(Action(None, gripper_cmd, np.eye(4), "Simple"))
```

---

## 任务配置示例分析

基于`left_place_cola_can_into_box_galbot_v2.json`：

### Place Stage配置

```json
{
    "action": "place",
    "active": {
        "object_id": "geniesim_2025_target_grasp_object",
        "primitive": null
    },
    "passive": {
        "object_id": "geniesim_2025_target_storage_box",
        "primitive": null
    },
    "extra_params": {
        "arm": "left",
        "place_with_origin_orientation": true,
        "disable_collision_links": ["left_gripper.*", "right_gripper.*"]
    }
}
```

**关键参数说明：**
- `active`: 被放置的物体（饮料瓶）
- `passive`: 目标容器（收纳盒）
- `arm`: 使用左臂
- `place_with_origin_orientation`: 保持物体原始朝向
- `disable_collision_links`: 禁用夹爪的碰撞检测

### 成功检查器

```json
"checker": [{
    "checker_name": "distance_to_target",
    "params": {
        "object_id": "geniesim_2025_target_grasp_object",
        "target_id": "geniesim_2025_target_storage_box",
        "target_offset": {
            "frame": "world",
            "position": [0, 0, -0.06]
        },
        "value": 0.12
    }
}]
```

验证物体是否成功放入收纳盒（距离容器底部6cm处，误差范围12cm）。

---

## 标注文件的作用

标注文件（如`benchmark_storage_box_001`的annotation）定义了：

1. **交互元素（interaction elements）**：
   - 放置点的位置和方向
   - 对齐方式（平移、旋转）
   - 采样数量

2. **几何信息**：
   - 容器的开口方向
   - 可放置区域的范围
   - 对称性信息

3. **约束条件**：
   - 允许的旋转角度
   - 位置偏移范围

这些信息被`get_aligned_pose()`使用，生成符合物理约束的候选位姿。

---

## 流程总结

```
1. 获取当前抓取状态（gripper2obj变换）
   ↓
2. 根据标注生成N个对齐位姿
   ↓
3. 转换为夹爪位姿
   ↓
4. 过滤倒置姿态
   ↓
5. 按中心距离排序，取前100个
   ↓
6. Simple IK预筛选
   ↓
7. AvoidObs IK完整求解
   ↓
8. （可选）生成并验证pre-place位姿
   ↓
9. 按人形化/关节距离排序
   ↓
10. 按中心距离二次排序（混合权重）
   ↓
11. 转换为规范坐标系
   ↓
12. 返回排序后的候选位姿列表
```

**设计理念：**
- **多阶段过滤**：从几何约束 → 运动学约束 → 优化排序
- **渐进式求解**：先快速筛选，再精确计算
- **鲁棒性保证**：近点搜索、噪声添加、多候选保留
- **任务导向**：中心距离优先，确保放置稳定性

---

## 关键函数速查

| 函数 | 作用 | 位置 |
|------|------|------|
| `select_pose()` | 主流程入口 | place.py:95 |
| `get_aligned_pose()` | 生成对齐位姿 | planner/common.py |
| `sort_place_pose_indices_by_center()` | 中心距离排序 | place.py:17 |
| `find_near_point_grasp_pose()` | 近点搜索 | place.py:43 |
| `sorted_by_position_humanlike()` | 人形化排序 | func/sort_pose/sort_pose.py |
| `sorted_by_joint_pos_dist()` | 关节距离排序 | func/common.py |
| `generate_action_sequence()` | 生成动作序列 | place.py:360 |
