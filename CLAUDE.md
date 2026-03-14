problem: in source/data_collection  pipeline-pick&place, i want to do visualization in pick&place stage-select_pose (we only care for pick now), every filter-step, what kind of pose they dropped; and after the ik-check passed, i want to show how motion-gen will really plan according to the sorted pose(so i can evaluate the pose-sorting algorithm's quality).

for example, before ik-check, you could visualize grasp by drawing a simple grasp like this "]-"(+x is approach-direction, +y is width-left direction, +z is top direction). for this section, you can use unit_lab/grasp_vis/grasp_viser_codex.py as reference, it's a test-labeled-grasp-pose-direction program, and we already have test result, so you don't need draw rgb-axises now. 

after ik checked, you could play motion-gen for each pose by the sorted order, and reset to the start stage. for this section, you can use isaacsim carb keyborad-interaction to "go next" and "reset".

again, because this task is complicated, you only care for the visualization(pose-filters, selection, motion_gen sorting). we don't need to finish move-grasp-place, just make it over at move.
