# Place 候选位姿生成链路重写

日期: 2026-04-03

本文档重写 `place` 阶段里“从容器的 `passive-place` 标注 + 被抓取物体的 `active-place` 标注，生成一批 place 候选位姿”的真实实现链路，重点补清：

- `get_aligned_pose()` 到底做了什么。
- 每一阶段的位姿、点、方向分别在哪个坐标系下。
- `select_pose()` 返回的 `grasp_pose` 实际上是什么。

本文基于当前工作区代码：

- `source/data_collection/client/planner/action/place.py`
- `source/data_collection/client/planner/common.py`
- `source/data_collection/client/layout/object.py`
- `source/data_collection/client/planner/action/stage.py`

本文只展开普通分支 `get_aligned_pose()`。
`passive_obj_id` 含 `"fix_pose"` 时会走 `get_aligned_fix_pose()`，那是另一条特殊分支。

## 1. 先统一记号和坐标系

下面统一用 `^X T_Y` 表示“`Y` 坐标系相对 `X` 坐标系的 4x4 位姿矩阵”。

### 坐标系

- `W`: 世界坐标系。
- `A`: 被放置物体 active object 的局部坐标系。
- `P`: 容器/目标物体 passive object 的局部坐标系。
- `G`: 当前抓着物体的 gripper/末端执行器坐标系。

### 当前实时位姿

在 `PlaceStage.select_pose()` 一开始：

- `^W T_A_cur = objects[self.active_obj_id].obj_pose`
- `^W T_G_cur = robot.get_ee_pose(...)`
- `^W T_P = objects[self.passive_obj_id].obj_pose`

代码里还有一个变量：

```python
gripper2obj = np.linalg.inv(object_pose) @ ee_pose
```

按矩阵语义，它其实是：

```text
^A T_G = (^W T_A_cur)^(-1) @ ^W T_G_cur
```

也就是“当前 gripper 在 active object 局部坐标系下的位姿”。
变量名叫 `gripper2obj`，但它实际保存的不是“gripper 到 object”的逆，而是 `object-local -> gripper` 这个相对关系。

这个相对关系会在后面被保留下来：

```text
^W T_G_target = ^W T_A_target @ ^A T_G
```

意思是：只要目标物体位姿定了，目标 gripper 位姿就由当前抓取关系直接推出。

## 2. active-place / passive-place 标注各自是什么

`update_aligned_info()` 会把 annotation 里的字段写回 object：

```python
self.xyz = element["xyz"]
self.direction = element["direction"]
self.constraint_axis = element.get("constraint_axis", ...)
self.angle_sample_num = element.get("angle_sample_num", 72)
```

这里最重要的一点是：

- `active_element["xyz"]` 在 `A` 系下。
- `active_element["direction"]` 在 `A` 系下。
- `active_element["constraint_axis"]` 在 `A` 系下。
- `passive_element["xyz"]` 在 `P` 系下。
- `passive_element["direction"]` 在 `P` 系下。
- `passive_element["constraint_axis"]` 在 `P` 系下。

也就是说，annotation 本身不是世界系数据，而是“各自物体局部系里的交互标注”。

## 3. `format_object()` 先把 annotation 变成一根短箭头

`get_aligned_pose()` 的第一步不是直接对齐 `xyz` 和 `direction`，而是先调用：

```python
active_object = format_object(active_obj, type="active", distance=0.01)
passive_object = format_object(passive_obj, type="passive", distance=0.01)
```

这里 `distance=0.01` 很关键。代码会先把 `direction` 归一化，再缩放到固定长度 `0.01m`。

这意味着：

- `direction` 的长度本身不参与 place 候选生成。
- place 只看 `direction` 的方向，不看它原始模长。

### 3.1 active 标注在 `A` 系下如何变成箭头

对 active object：

```text
^A p_a_start = active.xyz
^A d_a = normalize(active.direction)
^A p_a_end = ^A p_a_start + 0.01 * ^A d_a
```

也就是：

- active 的 `xyz` 被当成箭头起点。
- active 的 `direction` 指向箭头终点。

### 3.2 passive 标注在 `P` 系下如何变成箭头

对 passive object：

```text
^P p_p_end = passive.xyz
^P d_p = normalize(passive.direction)
^P p_p_start = ^P p_p_end - 0.01 * ^P d_p
```

注意这里和 active 刚好相反：

- passive 的 `xyz` 被当成箭头终点。
- passive 的箭头从 `xyz - 0.01 * direction` 指向 `xyz`。

这一步非常关键，因为后面平移是让：

```text
active 的 xyz_start  对齐到  passive 的 xyz_end
```

也就是说，从代码实现看：

- active 的 `xyz` 是“被送去对齐的源点”。
- passive 的 `xyz` 是“希望 active 最终落到的目标点”。

## 4. `obj2world()` 把箭头和约束轴都变到世界系

`format_object()` 产出的点和轴还在各自物体局部系中。
接下来 `obj2world()` 会用当前物体 pose 把它们变到 `W` 系。

### 4.1 点

对 active：

```text
^W p_a_start = ^W T_A_cur * ^A p_a_start
^W p_a_end   = ^W T_A_cur * ^A p_a_end
```

对 passive：

```text
^W p_p_start = ^W T_P * ^P p_p_start
^W p_p_end   = ^W T_P * ^P p_p_end
```

### 4.2 方向轴

世界系下的箭头方向是：

```text
^W d_a = normalize(^W p_a_end - ^W p_a_start)
^W d_p = normalize(^W p_p_end - ^W p_p_start)
```

如果 annotation 提供了 `constraint_axis`，代码还会把它乘当前物体旋转部分，得到：

```text
^W c_a = R(^W T_A_cur) @ normalize(^A c_a)
^W c_p = R(^W T_P) @ normalize(^P c_p)
```

所以到了 `get_aligned_pose()` 真正求解时，参与计算的输入都已经在世界系里了。

## 5. `get_aligned_pose()` 先求一个“基础对齐位姿”

这一步是整个 place 候选生成的核心。

### 5.1 先求基础旋转 `R0`

`get_aligned_pose()` 有两种情况。

#### 情况 A: active/passive 两边都有合法 `constraint_axis`

调用：

```python
R = calculate_rotation_from_two_axes(
    active_obj_world["direction"],
    a_cons,
    passive_obj_world["direction"],
    p_cons,
)
```

这相当于同时满足两件事：

- 让 active 的主方向 `^W d_a` 对齐到 passive 的主方向 `^W d_p`。
- 再用 `constraint_axis` 决定“绕主方向的零相位”。

更直白地说：

- `direction` 解决“朝哪边”。
- `constraint_axis` 解决“绕这根轴转到哪个角度算 0 度”。

#### 情况 B: 其中一边没有可用 `constraint_axis`

调用：

```python
R = calculate_rotation_matrix(^W d_a, ^W d_p)
```

这时只保证主方向对齐，不额外固定绕主方向的相位。

### 5.2 再求基础平移 `t0`

代码是：

```python
T = passive_obj_world["xyz_end"] - R @ active_obj_world["xyz_start"]
```

对应到记号：

```text
t0 = ^W p_p_end - R0 @ ^W p_a_start
```

这里的含义非常直接：

- 把 active 标注点 `^W p_a_start`
- 先按 `R0` 旋转
- 再平移到 passive 标注点 `^W p_p_end`

于是构造出一个世界系刚体变换：

```text
^W ΔT_align = [R0, t0]
```

再左乘到当前 active object pose：

```text
^W T_A_base = ^W ΔT_align @ ^W T_A_cur
```

`^W T_A_base` 就是“基础对齐后的 active object 世界位姿”。

### 5.3 这一步到底保证了什么

经过这一步，代码保证的是：

```text
1. active 的 place 标注点 被送到 passive 的 place 标注点
2. active 的 place 方向 对齐到 passive 的 place 方向
3. 如果提供 constraint_axis，则 0 度相位也被固定
```

这是第一批 place 候选的“母位姿”。

## 6. 然后绕 passive 的放置轴做整圈离散采样

基础位姿只给出一个对齐结果。
真正的候选集合来自后面的绕轴采样。

### 6.1 采样数 `N_align`

在 `place.py` 里：

```python
N_align = np.lcm(passive_obj.angle_sample_num, active_obj.angle_sample_num)
```

也就是：

```text
N_align = lcm(passive 角采样数, active 角采样数)
```

从实现上看，它被当成“绕 place 轴一圈要离散成多少份”的采样数。

### 6.2 旋转轴在哪个坐标系

后面调用：

```python
rotate_around_axis(
    target_obj_pose,
    passive_obj_world["xyz_start"],
    passive_obj_world["direction"],
    angle,
)
```

这里旋转轴完全定义在世界系：

- 轴上一点是 `^W p_p_start`
- 轴方向是 `^W d_p`

因为 `^W p_p_end` 也在同一条直线上，所以你也可以把它理解成：

- “绕 passive place annotation 那条世界系轴线转”

### 6.3 每个采样角的候选物体位姿

对第 `i` 个采样角：

```text
angle_i = i * 360 / N_align
^W T_A_i = Rot_W(axis = line(^W p_p_start, ^W d_p), angle_i) @ ^W T_A_base
```

因此 `get_aligned_pose()` 的返回值是：

```text
[
  ^W T_A_0,
  ^W T_A_1,
  ...,
  ^W T_A_(N_align-1)
]
```

全部都是“active object 在世界系下的候选放置位姿”。

### 6.4 `constraint_axis` 在这一步的真实作用

很多人会误以为有了 `constraint_axis` 就不需要绕轴采样。
当前实现不是这样。

真实逻辑是：

- `constraint_axis` 只负责决定基础位姿 `^W T_A_base` 的“零相位”。
- 然后代码仍然会围绕 passive 轴做一整圈离散采样。

所以：

- `constraint_axis` 决定“从哪里开始转”。
- `angle_sample_num` / `N_align` 决定“转多少个候选”。

## 7. 从候选物体位姿，变成候选 gripper 位姿

`get_aligned_pose()` 返回的是 world-frame object poses，不是 gripper poses。

在 `place.py` 里：

```python
target_gripper_poses = target_obj_poses @ gripper2obj[np.newaxis, ...]
```

按记号写就是：

```text
^W T_G_i = ^W T_A_i @ ^A T_G
```

这一步仍然在世界系下完成，输出是：

- 一批候选 `^W T_G_i`
- 每个候选都保持“当前抓住这个物体时的相对抓取关系不变”

## 8. 后续过滤、IK、排序时，各自用的是什么坐标系

### 8.1 upside-down 过滤

过滤直接看 `target_gripper_poses` 的旋转矩阵列向量。

也就是：

- 判断对象是基于 `^W T_G_i` 的旋转部分
- 本质上是世界系下的朝向检查

### 8.2 第一次“中心排序”

代码名叫 `sort_place_pose_indices_by_center()`，但它实际用的是：

```python
passive_center = passive_obj_pose[:3, 3]
center_dist = np.linalg.norm(target_obj_poses[:, :3, 3] - passive_center, axis=1)
```

所以它比较的是：

- candidate active object 原点在世界系中的位置 `^W T_A_i[:3,3]`
- 与 passive object 原点 `^W T_P[:3,3]`

注意：

- 这里不是 passive annotation 点 `^W p_p_end`
- 也不是容器网格几何中心
- 而是 `passive_obj.obj_pose` 的平移部分

如果某个物体局部原点恰好不在几何中心，这个“center 排序”就不是真正的几何中心排序。

### 8.3 IK 阶段

Simple IK 和 AvoidObs IK 输入的都是：

```text
^W T_G_i
```

也就是候选 gripper 在世界系下的目标位姿。

这两个 IK 阶段不会直接改 annotation；它们只是在世界系下筛掉机器人到不了、或者避障失败的 gripper 目标。

### 8.4 `use_near_point` 的近点搜索

如果 AvoidObs IK 全失败，代码会直接改：

```python
new_target_gripper_poses[:, 0/1/2, 3] += offset
```

这说明近点搜索偏移发生在：

- 候选 gripper 的世界系平移分量上

也就是：

- 沿 `W` 系的 `x/y/z` 方向找近点
- 不是沿 `A` 系，也不是沿 `P` 系

### 8.5 `use_pre_place` / `pre_insert_pose`

这一段最容易把坐标系看混。

### 先把 surviving world object pose 转成 passive-local pose

代码：

```python
target_obj_pose_canonical = np.linalg.inv(anchor_pose) @ target_obj_pose
```

也就是：

```text
^P T_A_i = (^W T_P)^(-1) @ ^W T_A_i
```

因此 `target_obj_pose_canonical` 不是世界系，而是：

- active object 相对 passive object 的位姿
- 即 `P` 系下的 active object pose

### 再在 `P` 系下往“pre-insert 方向”退一点

代码：

```python
normal_direction = np.array(self.passive_element[0]["direction"])
pre_insert_pose_canonical[:, :3, 3] += -normal_direction * self.pre_insert_offset
```

这里有两个关键点：

1. `normal_direction` 直接取自 `passive_element["direction"]`，所以它在 `P` 系下。
2. 代码这里没有再次归一化 `normal_direction`。

因此从实现上看，它隐含假设：

- passive annotation 里的 `direction` 已经是单位向量，或者至少模长可接受。

否则：

- `pre_insert_offset` 的真实位移长度会被 `direction` 的模长一起放大或缩小。

随后又把它变回世界系 object pose，再变成世界系 gripper pose 做 IK：

```text
^W T_A_pre_i = ^W T_P @ ^P T_A_pre_i
^W T_G_pre_i = ^W T_A_pre_i @ ^A T_G
```

### 这组 `pre_insert_pose` 最终是什么坐标系

最终保留下来的 `pre_insert_pose` 仍然是：

```text
^P T_A_pre_i
```

也就是 passive local frame 下的 active object pose。

补充一点：

- `PlaceStage.select_pose()` 确实会产出 `pre_insert_pose`
- 但普通 `PlaceStage.generate_action_sequence(self, grasp_pose)` 不接这个参数
- 真正消费 `pre_insert_pose` 的是 `InsertStage.generate_action_sequence(self, grasp_pose, pre_insert_pose=None)`

所以从当前代码看，`pre_insert_pose` 主要是给 `insert` 用的，不是普通 `place` 默认会执行的步骤。

### 8.6 最终排序

最终排序有两层：

### 第一层

- G2: 按人形化姿态排序
- 其他机器人: 按关节距离和雅可比分数排序

这些排序的输入主要来自 IK 解和 link pose，属于机器人求解空间，不是 annotation 空间。

### 第二层

又调用一次：

```python
sort_place_pose_indices_by_center(
    target_obj_pose_pass_ik,
    anchor_pose,
    base_idx_sorted=idx_sorted,
    center_weight=...
)
```

这里的 `target_obj_pose_pass_ik` 仍是：

```text
^W T_A_i
```

所以最后这次 center re-rank 仍然是在世界系里，用 active object 原点到 passive object 原点的距离做混合重排。

## 9. `select_pose()` 最后返回的到底是什么

最后代码做了：

```python
target_obj_pose_canonical_sorted = np.linalg.inv(anchor_pose) @ target_obj_pose_sorted
tmp_result["grasp_pose"] = target_obj_pose_canonical_sorted[i]
```

对应到记号：

```text
result[i]["grasp_pose"] = ^P T_A_i
```

这意味着返回值里的 `grasp_pose`：

- 不是 gripper pose
- 不是世界系 object pose
- 而是 passive local frame 下的 active object target pose

这是当前 place 文档最容易写错的一点。

如果开启 pre-insert，还会返回：

```text
result[i]["pre_insert_pose"] = ^P T_A_pre_i
```

同样也是 passive local frame。

## 10. 后续执行阶段如何把它重新变回世界系

`Stage.solve_target_gripper_pose()` 会把 canonical pose 还原成世界系目标：

```python
target_pose = anchor_pose @ target_pose_canonical
target_pose = transform_world @ target_pose
```

也就是：

```text
^W T_A_goal = ^W T_offset @ ^W T_P @ ^P T_A_goal
```

其中：

- `^P T_A_goal` 就是上一步返回的 `grasp_pose`
- `^W T_P` 是 passive object 当前世界位姿
- `^W T_offset` 是额外附加的世界系变换

### 10.1 普通 place 的第一步动作

`PlaceStage.generate_action_sequence()` 默认第一步：

```python
place_transform_up[:3, 3] = [0, 0, 0.05]
Action(target_pose_canonical, None, place_transform_up, "AvoidObs")
```

因此这里的 `place_transform_up` 是：

- 世界系下的上移 `+5cm`
- 不是 passive local frame 下的上移

也就是执行目标实际变成：

```text
^W T_A_goal_above = ^W T_up(世界系 +z 5cm) @ ^W T_P @ ^P T_A_goal
```

再根据当前抓取关系恢复 gripper 目标：

```text
^W T_G_goal_above = ^W T_A_goal_above @ ^A T_G
```

### 10.2 `post_place_action` 的方向又在哪个系

`post_place_action` 里，代码直接改的是：

```python
target_pose_canonical[:3, 3] += post_place_direction * post_place_distance
```

这里 `target_pose_canonical` 就是 `^P T_A`。
所以：

- `post_place_direction` 也是按 `P` 系解释的
- 这和 `place_transform_up` 的世界系偏移不是一个坐标系

当前实现里，这两个偏移来源不同：

- `place_transform_up`: 世界系
- `post_place_direction`: passive local frame

## 11. 把整条链路压成一条公式

如果只看普通 `get_aligned_pose()` 分支，place 候选的主链路可以压成下面这几步：

### 步骤 1: 从 annotation 构造局部箭头

```text
active:
  ^A p_a_start = active.xyz
  ^A p_a_end   = active.xyz + 0.01 * normalize(active.direction)

passive:
  ^P p_p_start = passive.xyz - 0.01 * normalize(passive.direction)
  ^P p_p_end   = passive.xyz
```

### 步骤 2: 变到世界系

```text
^W p_a_start, ^W p_a_end, ^W d_a, ^W c_a
^W p_p_start, ^W p_p_end, ^W d_p, ^W c_p
```

### 步骤 3: 求基础对齐位姿

```text
R0 = align((^W d_a, ^W c_a), (^W d_p, ^W c_p))
t0 = ^W p_p_end - R0 @ ^W p_a_start
^W T_A_base = [R0, t0] @ ^W T_A_cur
```

### 步骤 4: 绕 passive 轴离散采样

```text
^W T_A_i = Rot(axis = line(^W p_p_start, ^W d_p), angle_i) @ ^W T_A_base
```

### 步骤 5: 保持当前抓取关系，得到 gripper 候选

```text
^A T_G = (^W T_A_cur)^(-1) @ ^W T_G_cur
^W T_G_i = ^W T_A_i @ ^A T_G
```

### 步骤 6: IK / 排序后，转成 passive local canonical pose 输出

```text
^P T_A_i = (^W T_P)^(-1) @ ^W T_A_i
result[i]["grasp_pose"] = ^P T_A_i
```

## 12. 最容易搞错的点

### 误区 1: `get_aligned_pose()` 返回 gripper pose

不是。
它返回的是 active object 的世界系候选位姿 `^W T_A_i`。

### 误区 2: place annotation 直接在世界系下对齐

不是。
annotation 一开始都在各自物体局部系里，先经 `obj_pose` 变到世界系，再参与对齐。

### 误区 3: active/passive 的 `xyz` 语义相同

不是。

- active 的 `xyz` 被当成 source point
- passive 的 `xyz` 被当成 target point

这是 `format_object()` 明确编码出来的。

### 误区 4: 返回结果里的 `grasp_pose` 是抓手 pose

不是。
它是 `^P T_A_target`，即 passive local frame 下的目标物体 pose。

### 误区 5: `sort_place_pose_indices_by_center()` 用的是 passive 标注点

不是。
它用的是 `passive_obj.obj_pose[:3,3]`，即 passive object 原点在世界系中的位置。

### 误区 6: `place_transform_up` 也是 passive local offset

不是。
它是一个世界系偏移矩阵。

### 误区 7: `place_with_origin_orientation` 是在 `PlaceStage.select_pose()` 里生效的

不是。

从当前代码看，`place_with_origin_orientation` 主要在 `GraspStage` 里被读取，用来在“下一阶段是 place-like action”时影响抓取姿态筛选；它不是这篇文档展开的 `get_aligned_pose()` 主链路里的一个开关。

## 13. 可以直接对照代码看的最短路线

如果只想顺着代码快速核对一遍，建议按这个顺序看：

1. `source/data_collection/client/planner/action/place.py`
   `PlaceStage.select_pose()`
2. `source/data_collection/client/layout/object.py`
   `update_aligned_info()`
3. `source/data_collection/client/planner/common.py`
   `format_object()`
4. `source/data_collection/client/planner/common.py`
   `obj2world()`
5. `source/data_collection/client/planner/common.py`
   `get_aligned_pose()`
6. `source/data_collection/client/planner/action/stage.py`
   `solve_target_gripper_pose()`

如果只记一句话，可以记成：

```text
active-place 和 passive-place 都先在各自物体局部系里定义；
get_aligned_pose() 先把它们变到世界系做“点对点 + 轴对轴”的基础对齐，
再绕 passive 的 place 轴做整圈离散采样；
最后把 surviving 的 world object pose 再转回 passive local frame 输出。
```
