# SingleTask --> MetaTask  data_collection pipeline升级

## 参考:
1. source/data_collection/tasks/diy/meta_task/meta_pick_place_V0.json
2. source/data_collection/tasks/diy/single_task/left_place_apple_into_box_galbot_v1.json
3. source/data_collection/scripts/data_collector_server.py
4. source/data_collection/scripts/run_data_collection.py


## Goal:
现有流程只能设置单一task-json 多episodes采集
我希望更改流程 使得可以一次性配置多个task-json 和 各自运行的episodes数

1) 一个Meta Task里，并不会换房间级场景, 但是被抓取的目标candidates、放置目标candidates等都可能变
2) 先有流程会生成saved_task下的具体配置文件，对于MetaTask流程，我希望每个single task执行完所有的episodes 切换到next-task时，才会对next-task生成saved_task下的具体配置文件，而不是在一开始就生成所有task的具体文件
3) 我希望最终结束后能有一个统计，关于每个task具体成功生成了多少个episodes，以一个json形式的文件存储在meta_task下并运行中不断更新(考虑运行可能终端，每个single_task成功一个episode就更新一次)

## Notes:
上述为基本设置,你在查看具体代码和制定plan时，可能发现实现上的问题，可以多与我进行讨论决策
