# Noise Explorer

用于离线查看 `source/data_collection/server/ros_publisher/camera_noiser.py` 里的图像加噪效果。

当前目录内容：

- `head_color_first_frame.png`
  来自 `source/data_collection/recording_data/place_cola_can_into_blue_box_galbot_v1_0402_2108_1/observations/videos/head_color.mp4` 的第一帧。
- `noise_models.py`
  纯 CPU 的噪声实现，模式和参数名与 `camera_noiser.py` 对齐。
- `noise_explorer.py`
  本地交互 UI。

## Run

```bash
python unit_lab/noiser/noise_explorer.py
```

可选参数：

```bash
python unit_lab/noiser/noise_explorer.py --image /path/to/another_image.png
python unit_lab/noiser/noise_explorer.py --smoke-test
python unit_lab/noiser/noise_explorer.py --export-grid unit_lab/noiser/default_grid.png
```

## What The UI Shows

- 左侧可以切换噪声模式、修改参数、固定或随机化 seed。
- 中间会显示原图和当前噪声结果。
- 下方总览会同时显示全部噪声模式，方便横向比较。
- 每个参数下面会同时展示：
  - `doc`：文件头注释里的说明范围
  - `current`：`get_random_parameters(...)` 现在真正会采样的范围

## Notes

- `gaussian / salt_pepper / poisson / speckle / quantization` 是当前 `publish_noised_rgb` 会实际使用的模式。
- `sensor_noise / brownian` 在 `camera_noiser.py` 中有实现，但当前发布路径没有接入。
- `gaussian` 预览保留了当前 Warp 版本的 `uint8` 直接转换行为，因此极端值会出现 wrap-around，而不是 clip。
