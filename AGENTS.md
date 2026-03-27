# Current Task
## Reference Files
1. unit_lab/grasp_vis/interaction_pose_browser.py
this is a pose visualizer for assets under /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects, the labeled operation-poses of assets(that do have labeled operation-poses) are under /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction.
so for those assets that don't have label now, we need to label them.

2. GraspGen as grasp-pose label tool:
/home/agxi/ManipLab/GraspGen/client-server/  ......
/home/agxi/ManipLab/GraspGen/scripts/demo_object_mesh.py
a specific example would be: 
"python scripts/demo_object_mesh.py --mesh_file /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/bottle/benchmark_bottle_017/Aligned.usda --mesh_scale 1.0 --gripper_config /home/agxi/GraspGen/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml --output_file ./box_grasps.yml --num_grasps 50"


## Plan
1) make a new interactable visualize-and-edit program under unit_lab/grasp_vis/
2) use GraspGen to create grasp-poses by server--client pipeline
3) first, we make grasp-labels for assets that already been labeled, so we can compare with existed labels with the ones GraspGen generated, to ensure they are axis-aligned and offset-proper
4) for place poses, we set them manually by isaacsim-ui interaction


