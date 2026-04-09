references:
1) source/data_collection/client/planner/action/place.py
2) docs/place_pose_generation_flow_codex.md 之前的agent总结的Place位姿策略
3) 正常运行，成功率高的 place-container-object-asset 
物体路径 /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/storage_box/benchmark_storage_box_007
标注位姿路径 /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_storage_box_007
4) 几乎place必失败，触发如下报错的 place-container-object-asset:
2026-04-09T13:05:18Z [205,128ms] INFO     [stage.py:354] Unable to find valid target_obj_pose for place action, try next active/passive element combination.
2026-04-09T13:05:18Z [205,128ms] WARNING  [run_data_collection.py:381] Stage 1 place initialize action sequence buffer failed

物体路径:
/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/objects/benchmark/web_dish/benchmark_web_dish_0079
标注位姿路径:
/home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/interaction/benchmark_web_dish_0079

我觉得我标注的 passive-place 位姿并没有什么问题，为什么无法通过?
