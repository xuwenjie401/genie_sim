# Galbot Pick Pose 筛选补充方案与 Coding Plan（agent-A）

日期：2026-04-05

Problem：
- 当前 `pick` 阶段的抓取位姿筛选还缺少两类约束：
  - 对于 galbot，夹爪前进方向 `direction_approaching`，简称 `da`，定义为夹爪局部 `x` 轴方向。
  - 需要删去 `da` 近似竖直向下的抓取位姿，例如“从正上方往下抓”；示例阈值为 `15 deg`。
  - 需要删去“从机器人前方朝向机器人”的抓取位姿，也就是从远往近抓的姿态。
- 用户还要求分析当前 galbot 的抓取位姿排序逻辑，并给出更细的 coding plan。
- 本文只输出设计与实现计划，不修改业务代码。

范围：
- 本文聚焦 `source/data_collection/client/planner/action/grasp.py` 中 `PickStage.select_pose()` 的筛选逻辑。
- “从远往近抓”的判断仍然在 robot base 系下进行，但 base 系直接按默认 `robot_init_pose` 解释，不单独为很小的 init 扰动做额外设计。
- 只对 galbot 增加该补充筛选。
- 本文同时记录当前 galbot 的排序现状，但本轮 plan 不包含排序策略改写。

相关文件：
- [grasp.py](../source/data_collection/client/planner/action/grasp.py)
- [common.py](../source/data_collection/client/planner/func/common.py)
- [omni_robot.py](../source/data_collection/client/robot/omni_robot.py)
- [left_place_cola_can_into_box_galbot_v1.json](../source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json)

## Agent-A 执行摘要

- 推荐在 [`PickStage.select_pose()`](../source/data_collection/client/planner/action/grasp.py#L44) 中新增一个 galbot-only 的 approach-direction filter，位置放在 `grasp_poses = obj_pose @ grasp_poses_canonical` 之后、第一次 `random_downsample` 与第一次 `Simple IK` 之前。
- 对 galbot，`robot_gripper_2_grasp_gripper` 当前是单位阵，因此 `da` 可以直接取 `grasp_pose[:3, 0]`，即抓取位姿的局部 `x` 轴在世界系中的方向。
- “近竖直”本次只处理“近似竖直向下”，不处理竖直向上的姿态；默认禁止，只有 task-json 显式写放开字段时才允许。
- “从远往近抓”必须在 robot base 系中判断，但 base 系直接按默认 `robot_init_pose` 解释即可；这个约束对 galbot 默认开启，不依赖 task-json 配置。
- 当前 galbot 的最终排序并不使用 `jacobian_score`，也不使用 `grasp pose direction`。过滤完成后，最终排序只按当前关节位形到目标 IK 解的 joint-space 距离升序。
- 这次 plan 不改排序，只补筛选；否则会把“候选质量控制”和“排序代价重定义”两个变化混在一起，回归面会明显变大。

## 现状分析

### 1. `da` 在当前代码里的轴定义

结论：
- 对 galbot，`da` 可以直接视为 `grasp_pose[:3, 0]`。

证据：
- `select_pose()` 会先将 `grasp_poses_canonical[:, :3, :3]` 右乘 `robot.robot_gripper_2_grasp_gripper`，见 [grasp.py#L75-L78](../source/data_collection/client/planner/action/grasp.py#L75-L78)。
- galbot 的 `robot_gripper_2_grasp_gripper` 当前为单位阵，见 [omni_robot.py#L73-L75](../source/data_collection/client/robot/omni_robot.py#L73-L75)。

因此：
- 对 galbot 来说，抓取候选的局部坐标轴不被额外重排。
- 局部 `x` 轴在世界系中的方向向量就是 `grasp_poses[:, :3, 0]`。

### 2. 当前 `select_pose()` 的筛选主流程

当前主流程如下：
1. 读取对象自带 grasp pose 与 width。
2. 按 grasp 数据中的 `[:, 1, 3]` 做 percentile 过滤。
3. 对姿态乘上 `robot_gripper_2_grasp_gripper`。
4. 可选执行 `set_grasp_vertical`、`flip_grasp`、`filter_grasp_pose`、`humanlike_filter`。
5. 若 `disable_upside_down=true`，执行机器人特定的“非倒置”过滤。
6. `random_downsample(..., 300)`。
7. 应用 `grasp_offset`。
8. 执行第一次 `Simple IK`。
9. 若后继 stage 是 `place` 等，会做“下一步 IK 可行性”前瞻过滤。
10. `random_downsample(..., 100)`。
11. 执行 `AvoidObs IK`。
12. 依据排序函数输出最终 grasp 顺序。

关键位置：
- 当前最自然的新增筛选插入点是步骤 4 和步骤 6 之间，即 [grasp.py#L102-L138](../source/data_collection/client/planner/action/grasp.py#L102-L138)。

原因：
- 这一段仍处于“几何/姿态筛选”的阶段。
- 这时 `grasp_poses` 已经在世界系中，适合直接算 `da_world`。
- 这时还没开始随机下采样和 IK，能尽早减少无效候选。

### 3. 当前 galbot 的排序逻辑

结论：
- 当前 galbot 的排序不是“按 approach 方向偏好排序”，而是“先过滤，再按 joint movement cost 排序”。

分解如下：
- 第一道过滤：task 中配置了 `grasp_upper_percentile: 75`，见 [left_place_cola_can_into_box_galbot_v1.json#L277-L284](../source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json#L277-L284)。
- 第二道过滤：`disable_upside_down=true`，对 galbot 使用 `grasp_poses[:, 2, 2] > 0.0` 保留局部 `z` 轴朝上的候选，见 [grasp.py#L121-L128](../source/data_collection/client/planner/action/grasp.py#L121-L128)。
- 第三道过滤：如果后继 stage 是 `place`，代码会统计每个 grasp pose 在目标放置姿态集合上的 `Simple IK` 成功次数，只保留不低于 `max(median/2, 1)` 的候选，见 [grasp.py#L230-L327](../source/data_collection/client/planner/action/grasp.py#L230-L327)。
- 最终排序：调用 `sorted_by_joint_pos_dist_and_grasp_pose()`，见 [grasp.py#L482-L491](../source/data_collection/client/planner/action/grasp.py#L482-L491)。

但该排序函数的当前真实行为是：
- 它会计算当前 joint state 到每个 IK 解的欧氏距离。
- 它会对 `ik_jacobian_score` 做归一化。
- 但最终 `cost = joint_pos_dist`，没有使用 `ik_jacobian_score`，也没有使用 `grasp_poses` 或 `pre_grasp_offset`，见 [common.py#L123-L144](../source/data_collection/client/planner/func/common.py#L123-L144)。

因此：
- 当前 galbot 的最终排序只看“关节动得少不少”。
- `sorted_by_joint_pos_dist_and_grasp_pose()` 这个函数名对现状有误导性。

### 4. 当前代码里另一个相关但独立的问题

结论：
- 当前 `grasp_offset` 和 `pre_grasp` 的推进轴仍然按局部 `z` 轴处理，不是按 galbot 的 `da=x` 处理。

证据：
- `grasp_offset` 的平移方向来自 `grasp_rotate @ [0, 0, 1, 0]`，见 [grasp.py#L140-L147](../source/data_collection/client/planner/action/grasp.py#L140-L147)。
- `pre_grasp` 也是沿局部 `-z` 平移，见 [grasp.py#L444-L449](../source/data_collection/client/planner/action/grasp.py#L444-L449)。

说明：
- 这与本次新增筛选不是同一个问题。
- 但要注意：即便这次把筛选改成按 `da=x`，后续“接近物体”的 motion 语义仍不完全一致。
- 建议在本轮文档中记录为 follow-up，不与本次筛选改动绑在一起。

## 本次改动目标

### 功能目标

- 对 galbot 的 `pick` 阶段补上两个姿态筛选约束：
  - 默认删除 `da` 近似竖直向下的候选。
  - 默认删除在 robot base 系下“从机器人前方向机器人抓”的候选。

### 非目标

- 本次不改 `place` 逻辑。
- 本次不改排序 cost。
- 本次不统一抽象到所有机器人。
- 本次不改 `grasp_offset` / `pre_grasp` 的 approach 轴定义。

## 设计决策

### 1. 为什么第二个约束仍然要在 base 系判断

原因：
- “从机器人前方朝向机器人抓”是一个相对机器人朝向的语义，不是绝对世界方向语义。
- 不能假设所有任务 world 系都与 robot 前向严格对齐，见 [left_place_cola_can_into_box_galbot_v1.json#L171-L189](../source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json#L171-L189)。

因此：
- 必须先得到 `da_world`，再转换为 `da_base`。
- 对 galbot，只在 `if "galbot" in robot.robot_cfg.lower()` 下解释 `base +x` 为“前方”。
- base 系的解释直接按默认 `robot_init_pose` 即可，不额外设计去吸收很小的 init 扰动。

### 2. 为什么新增 helper 比复用 `filter_grasp_pose_by_gripper_up_direction()` 更合适

原因：
- 现有 `filter_grasp_pose_by_gripper_up_direction()` 的语义是“与某个固定方向夹角小于阈值则保留”，见 [common.py#L203-L280](../source/data_collection/client/planner/func/common.py#L203-L280)。
- 这不适合表达：
  - “只删除近似竖直向下”这种单侧约束。
  - “在 robot base 系里按前向符号过滤”这种相对坐标系约束。

建议：
- 在 `source/data_collection/client/planner/func/common.py` 新增专用 helper。
- `select_pose()` 只负责读取 `extra_params`、构造参数并调用 helper。

### 3. “近竖直向下”推荐采用的默认语义

推荐默认语义：
- 只删除近似竖直向下的候选，不删除近似竖直向上的候选。

即：
- `vertical_cos = cos(theta_vertical_deg)`
- `reject_vertical_down = da_world_z <= -vertical_cos`

理由：
- 用户已明确收紧语义，只需要控制“竖直往下”的部分。
- 这比之前的双侧约束更贴合需求，也能减少误删。

配置策略建议：
- 默认不允许近似竖直向下抓取。
- 如果某个 task 后续确实需要放开，再在 task-json 中显式写允许字段。
- 第一版不建议暴露复杂 mode 参数。

### 4. “从前方向机器人抓”推荐采用的判据

记：
- `da_world = grasp_poses[:, :3, 0]`
- `R_base_world = quat2mat_wxyz(robot.init_rotation)` 或等价默认 base 旋转矩阵
- `da_base = R_base_world.T @ da_world`

对于 galbot：
- 规定 `base +x` 为机器人前向。
- 若 `da_base_x < 0`，表示 approach 朝向机器人自身，即“从前方往近处抓”，应删除。
- 若 `da_base_x > 0`，表示 approach 朝离开机器人方向，即“从近往远抓”，允许保留。

建议保留一个数值 margin：
- 不直接用 `0`，而是用一个很小的容忍带，例如 `approach_base_x_min = 0.0` 或 `0.05`。
- 第一版默认直接启用该过滤，且阈值先用 `0.0`，保证语义最直接。

## 拟议参数设计

建议在 `pick` stage 的 `extra_params` 中只增加一个显式放开字段：

```json
"allow_galbot_top_down_grasp": true
```

说明：
- `allow_galbot_top_down_grasp`
  - 仅控制是否允许“近似竖直向下抓取”。
  - 默认不存在时按 `False` 处理，也就是默认不允许。
- “从远往近抓”不需要 task-json 开关，作为 galbot 默认开启的筛选约束直接生效。
- `15 deg` 阈值建议先写在代码默认值里；若后续确实需要调，再单独决定是否继续暴露阈值字段。

## Coding Plan

### Phase 1. 新增 helper，封装 galbot approach filter

目标：
- 在 `source/data_collection/client/planner/func/common.py` 新增一个专用 helper，例如：
  - `filter_galbot_grasp_pose_by_approach_direction(...)`

建议接口：

```python
def filter_galbot_grasp_pose_by_approach_direction(
    grasp_poses: np.ndarray,
    grasp_widths: np.ndarray,
    robot_base_rotation: np.ndarray,
    vertical_threshold_deg: float = 15.0,
    allow_top_down_grasp: bool = False,
    reject_towards_robot: bool = True,
    towards_robot_max_dot: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ...
```

输入说明：
- `grasp_poses`
  - 世界系下的抓取位姿，shape `(N, 4, 4)`。
- `grasp_widths`
  - 对应 width。
- `robot_base_rotation`
  - robot base 在世界系中的旋转矩阵。这里按默认 `robot_init_pose` 构造即可。

输出建议：
- `filtered_grasp_poses`
- `filtered_grasp_widths`
- `mask`
- `stats`
  - 例如记录：
    - `num_input`
    - `num_reject_vertical_down`
    - `num_reject_towards_robot`
    - `num_output`

内部计算步骤：
1. 取 `da_world = grasp_poses[:, :3, 0]`。
2. 计算近竖直向下判据：
   - `vertical_cos = np.cos(np.deg2rad(vertical_threshold_deg))`
   - 若 `allow_top_down_grasp` 为 `False`：
     - `reject_vertical = da_world[:, 2] <= -vertical_cos`
   - 若 `allow_top_down_grasp` 为 `True`：
     - `reject_vertical = np.zeros(len(grasp_poses), dtype=bool)`
3. 计算 base 系 approach：
   - `da_base = (robot_base_rotation.T @ da_world.T).T`
   - 要求在代码注释里说明矩阵方向，避免后续误改。
4. 计算“朝向机器人”判据：
   - 若 `reject_towards_robot` 为 `True`：
     - `reject_towards = da_base[:, 0] < towards_robot_max_dot`
   - 否则：
     - `reject_towards = np.zeros(len(grasp_poses), dtype=bool)`
5. 合并 mask：
   - `mask = ~reject_vertical & ~reject_towards`
6. 返回过滤结果和统计信息。

注意：
- helper 应该只处理 galbot 所需语义，不要做成过度抽象的“万能姿态筛选器”。
- `reject_towards_robot` 第一版虽然会默认传 `True`，但接口保留这个参数有利于调试。

### Phase 2. 在 `PickStage.select_pose()` 中接入 galbot-only 分支

目标：
- 在 `select_pose()` 中读取 `extra_params` 并调用 helper。

建议插入位置：
- 放在现有 `filter_grasp_pose` / `humanlike_filter` / `disable_upside_down` 之后，`random_downsample(..., 300)` 之前。

推荐顺序：
1. `filter_grasp_pose`
2. `humanlike_filter`
3. `disable_upside_down`
4. `filter_galbot_approach`
5. `random_downsample`

原因：
- `disable_upside_down` 已经是 robot-specific 的粗筛。
- 新增的 galbot filter 也是几何筛选，放在 downsample 之前最省 IK。
- 与后续 IK / next-stage 前瞻过滤逻辑解耦。

接入方式建议：

```python
allow_galbot_top_down_grasp = self.extra_params.get("allow_galbot_top_down_grasp", False)
if "galbot" in robot.robot_cfg.lower():
    ...
```

base 系旋转的获取建议：
- 直接使用默认 `robot_init_pose` 对应的 base 旋转构造矩阵。
- 不为小范围 init 扰动额外引入实时 base pose 依赖。

需要增加的日志：
- 在 filter 前后输出数量变化。
- 额外输出：
  - 删除了多少个近似竖直向下候选。
  - 删除了多少个朝向机器人候选。

日志示例：

```python
logger.info(
    f"{self.action_type}, {self.passive_obj_id}, "
    f"Filtered galbot approach poses: {stats['num_output']}/{stats['num_input']}, "
    f"vertical_down_reject={stats['num_reject_vertical_down']}, "
    f"towards_robot_reject={stats['num_reject_towards_robot']}"
)
```

### Phase 3. task-json 策略

目标：
- 不修改现有 `v1` task-json。
- 只在未来确实需要放开 top-down 抓取时，才在具体 task 中显式加字段。

默认行为：
- 对 galbot：
  - 默认禁止近似竖直向下抓取。
  - 默认禁止从远往近抓。
- 因此 [left_place_cola_can_into_box_galbot_v1.json](../source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json) 本轮不需要修改。

仅当某个 task 需要显式放开 top-down 时，再在 `pick.extra_params` 中增加：

```json
"allow_galbot_top_down_grasp": true
```

说明：
- 这个字段只在“需要放开默认禁止项”时才出现。
- 不建议把默认行为也写回每个 task-json，避免噪音配置。

### Phase 4. 做静态与运行验证

静态验证：
- 代码层面检查以下不变量：
  - helper 输入输出 shape 不变。
  - `grasp_poses` 与 `grasp_widths` 的 mask 始终同步。
  - 当所有候选都被删空时，`select_pose()` 能沿当前已有路径安全返回 `[]`。

运行验证建议：
1. 选取当前 task，固定随机种子或至少多跑数次 episode。
2. 在日志中确认：
   - galbot filter 已生效。
   - `vertical_down_reject` 与 `towards_robot_reject` 非零或至少数量合理。
3. 抽查最终返回的 top-k grasp pose：
   - `da_world_z > -cos(15°)`，除非 task 显式设置 `allow_galbot_top_down_grasp=true`。
   - `da_base_x >= 0`。
4. 确认没有明显增加：
   - `No grasp pose found`
   - `No best_grasp_poses can pass curobo IK`
   - `No grasp pose can pass next action IK`

建议增加的离线检查脚本或断言：
- 对最终 `result[:K]` 做打印或调试断言：
  - `da_world = grasp_pose[:3, 0]`
  - `da_base = R_base_world.T @ da_world`
  - 确认 `da_world[2] > -cos(threshold)`，除非允许 top-down。
  - 确认 `da_base[0] >= 0`。

## 验收标准

功能验收：
- 对 galbot：
  - 默认近似竖直向下候选被稳定删除；若 task 显式放开，则该项不再删除。
  - 在 base 系中朝向机器人的候选被稳定删除，且该约束默认开启。
- 对非 galbot：
  - 行为不变。

工程验收：
- 新增逻辑集中在 helper + `select_pose()` 接入点，代码结构清晰。
- 日志足够说明筛选效果。
- 不引入额外的排序语义变化。

数据侧验收：
- 采样结果里，明显的 top-down inward grasp 显著减少。
- 从远往近抓的候选显著减少。
- 任务成功率、可解率没有出现明显退化。

## 风险与注意事项

### 1. 15 度 top-down 阈值可能过严

风险：
- 对某些高瘦物体，如果 grasp 数据本来就偏 top-down，默认禁掉近似竖直向下抓取后，候选数可能下降较多。

缓解：
- 第一版先不暴露阈值。
- 仅通过 `allow_galbot_top_down_grasp=true` 作为第一层放开手段。
- 第一轮跑日志时重点关注删空比例。

### 2. 与当前排序逻辑的交互

说明：
- 这次新增筛选会改变候选集，因此 top-1 pose 很可能变化。
- 但这不是排序逻辑变了，而是候选空间先被裁掉了。

### 3. 与 `grasp_offset` / `pre_grasp` 的轴不一致

说明：
- 这次不修这个问题，但文档需要明确记录。
- 如果后续发现“筛掉了不合理 `da`，但 motion 还是沿 z 接近”导致行为别扭，应单独立项修复 approach axis。

## 最终建议

- 本次改动建议严格限定为“galbot-only 的候选筛选增强”，不要顺手改排序。
- 实现上应采用：
  - `select_pose()` 中 `if "galbot" in robot.robot_cfg.lower()` 的显式分支
  - 默认 base 系下判断“是否朝向机器人”
  - 默认点积阈值判断“是否近似竖直向下”
  - 仅通过 `allow_galbot_top_down_grasp` 这个 task 字段显式放开 top-down
- 这样改动最小，语义直接，对现有 pipeline 的侵入也最低。
