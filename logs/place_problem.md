  ---                                                                                                                                                                                                       
  How attach_obj works (the full chain)                                                                                                                                                                     
                                                                                                                                                                                                            
  1. Entry: command_controller.py:handle_attach_obj() (line 1056)                                                                                                                                           
                  
  Walks the USD stage, collects all UsdGeom.Mesh prim paths under the given object path, then calls ui_builder.attach_objs(items, is_right).

  2. ui_builder.py:attach_objs() (line 417)

  position, rotation_matrix = self._get_ee_pose(is_right, is_local=True)  # robot-local frame
  rotation = mat2quat_wxyz(rotation_matrix)
  result = curoboMotion.attach_obj(prim_path_list, link_name, position, rotation)
  Gets EE pose in the robot's local frame, then passes it down.

  3. motion_gen_reacher.py:attach_obj() (line 1020)

  self.motion_gen.detach_object_from_robot()  # clean slate
  self.set_obstacles()                          # rebuild static collision world from stage
  ee_pose = Pose(position=..., quaternion=...)
  self.attach_objects_to_robot(
      object_names=prim_paths,
      world_objects_pose_offset=Pose.from_list([0, 0, 0.005, 1, 0, 0, 0], ...),
      remove_obstacles_from_world_config=True,  # removes from static world
      ee_pose=ee_pose,
  )

  4. motion_gen_reacher.py:attach_objects_to_robot() (line 1048) — the bug is here

  if world_objects_pose_offset is not None:
      ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
  ee_pose = ee_pose.inverse()   # intended: gives ee_T_w transform
  # ...
  for object_name in object_names:
      obs = self.motion_gen.world_model.get_obstacle(object_name)
      sph = obs.get_bounding_spheres(n_spheres, surface_sphere_radius, ...)
      sph_list += [s.position + [s.radius] for s in sph]   # ← raw positions
      self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)  # disable as static

  spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))
  sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()
  self.motion_gen.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

  ---



Root Cause of Your Collision Bug

  ee_pose is computed but never applied to transform the sphere positions.

  CuRobo's attach_spheres_to_robot() expects sphere positions in the EE link's local frame. The correct transform needed is:

  sph_ee = T(ee←world) * T(world←obj) * sph_local

  What actually happens:
  - ee_pose (T(ee←world)) is computed at line 1061
  - obs.get_bounding_spheres() returns spheres in the obstacle's local/world frame
  - Those raw positions go directly into attach_spheres_to_robot — no frame transform applied

  So CuRobo receives sphere positions that are in world/object-local space but interprets them as EE-relative. As the arm moves during the post-grasp trajectory, those spheres stay "pinned" at their
  original world positions rather than tracking the actual bottle location. CuRobo plans around the wrong location — the bottle itself is not collision-avoided.

  Additionally there's a secondary issue: remove_obstacles_from_world_config=True removes the bottle from the static collision world (line 1084-1086), so the bottle's mesh doesn't block planning at all
  anymore, and the mislocated spheres don't compensate for it.

  ---
  Why arm/gripper is fine but bottle collides

  The robot arm and gripper have collision spheres defined in the URDF (attached_object link), and those are always correctly computed by CuRobo's FK. Only the attached object's spheres (from
  attach_spheres_to_robot) use this broken path — hence arm is fine, bottle penetrates.
