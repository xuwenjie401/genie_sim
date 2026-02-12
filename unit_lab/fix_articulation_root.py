import omni.kit.commands
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf

# ================= 配置区 =================
ROBOT_ROOT_PATH = "/galbot_one_golf"  # 你的机器人顶层路径
OLD_BASE_LINK_PATH = "/galbot_one_golf/base_link" # 之前的根链接
# ==========================================

stage = omni.usd.get_context().get_stage()

def fix_robot_articulation(root_path, old_base_path):
    print(f"--- 开始重构机器人物理结构: {root_path} ---")
    
    # 1. 彻底清除旧节点的 Articulation 属性
    old_base_prim = stage.GetPrimAtPath(old_base_path)
    if old_base_prim:
        apis_to_remove = [UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI]
        for api in apis_to_remove:
            if old_base_prim.HasAPI(api):
                omni.kit.commands.execute("RemovePhysicsAPI", prim_path=old_base_path, api=api)
                print(f"已移除旧的 API: {api.__name__} from {old_base_path}")

    # 2. 为顶层节点添加 Articulation Root
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim:
        print(f"错误: 找不到路径 {root_path}")
        return

    # 确保顶层没有 RigidBody（防止它自己参与碰撞）
    if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        omni.kit.commands.execute("RemovePhysicsAPI", prim_path=root_path, api=UsdPhysics.RigidBodyAPI)
        print(f"已移除顶层节点的 RigidBodyAPI (Articulation Root 不应有刚体属性)")

    # 添加 ArticulationRoot API
    omni.kit.commands.execute("ApplyArticulationRootAPI", prim_path=root_path)
    # 额外开启 Physx 的专业属性支持（可选但推荐）
    omni.kit.commands.execute("ApplyPhysxArticulationAPI", prim_path=root_path)
    
    # 3. 遍历并修复所有关节 (Joints)
    print("正在扫描并修复关节路径...")
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.Joint):
            joint_path = str(prim.GetPath())
            joint = UsdPhysics.Joint(prim)
            
            # 修复 body0 和 body1
            for rel_name in ["body0", "body1"]:
                rel = joint.GetRelationship(rel_name)
                targets = rel.GetTargets()
                
                if targets:
                    new_targets = []
                    for t in targets:
                        # 核心逻辑：确保路径是绝对路径且在当前 root 下
                        abs_path = t.MakeAbsolutePath(root_path)
                        new_targets.append(abs_path)
                        print(f"  [Joint] {prim.GetName()} -> {rel_name} 重定向至: {abs_path}")
                    
                    rel.SetTargets(new_targets)

    # 4. 解决可能的“爆炸”问题：禁用自碰撞（测试用）
    # 如果机器人依然散架/炸开，请手动在 UI 勾选 Articulation Root 下的 'Disable Self Collision'
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physx_art.CreateEnabledSelfCollisionsAttr().Set(False)

    print(f"--- 重构完成！请点击 Play 键测试 ---")

# 执行脚本
fix_robot_articulation(ROBOT_ROOT_PATH, OLD_BASE_LINK_PATH)
