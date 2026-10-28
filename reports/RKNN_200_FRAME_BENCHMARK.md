# RK3588 YOLO-World RKNN 200 帧性能测试

## 测试范围

- 测试日期：2026-09-22
- 设备：Orange Pi 5 Plus / RK3588
- RKNN Runtime：2.3.2
- RKNPU Driver：0.9.6
- 数据目录：`/home/orangepi/ros_bag/entire_floor_bag/out_put/dataset/results`
- 测试图片：`frame000000.jpg` 至 `frame000199.jpg`，共 200 帧
- 输入分辨率：640 × 640
- 每个模型预热：10 次
- Runtime core mask：`NPU_CORE_0_1_2`
- 推理方式：单进程、单 context、连续同步推理
- 延迟范围：仅统计 `RKNNLite.inference()`，不包括后处理、画框和图片保存
- NPU 负载接口：`/sys/kernel/debug/rknpu/load`
- 负载采样间隔：约 10 ms
- 空闲基线：Core0、Core1、Core2 均为 0%

## 测试模型

1. Rockchip 原始 FP16
   - `/home/orangepi/models/yolo_world_rknn/yolo_world_v2s_fp16.rknn`
   - 图像和 `1×80×512` 文本向量双输入
2. 静态类别 FP16
   - `/home/orangepi/models/test/yolov8s-worldv2_indoor_fp.rknn`
   - 12 个室内类别已写入模型
3. 静态类别 INT8
   - `/home/orangepi/models/test/yolov8s-worldv2_indoor_split_int8.rknn`
   - 12 个室内类别已写入模型

## 正式测试结果

| 模型 | 平均耗时 | 中位数 | P95 | 最小值 | 最大值 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Rockchip 原始 FP16 | 280.17 ms | 282.61 ms | 384.72 ms | 155.25 ms | 424.06 ms | 3.57 |
| 静态类别 FP16 | 172.10 ms | 148.59 ms | 280.94 ms | 112.14 ms | 336.40 ms | 5.81 |
| 静态类别 INT8 | 71.53 ms | 76.52 ms | 96.51 ms | 48.82 ms | 104.90 ms | 13.98 |

## NPU 逐核心负载

| 模型 | Core0 平均 | Core1 平均 | Core2 平均 | 三核负载之和 | 三核总容量占用 |
|---|---:|---:|---:|---:|---:|
| Rockchip 原始 FP16 | 47.21% | 3.86% | 3.86% | 54.92% | 18.31% |
| 静态类别 FP16 | 54.26% | 5.80% | 5.89% | 65.95% | 21.98% |
| 静态类别 INT8 | 79.37% | 9.80% | 9.80% | 98.96% | 32.99% |

其中“三核总容量占用”按以下方式计算：

```text
(Core0 + Core1 + Core2) / 3
```

三核负载之和 100% 相当于约一个 NPU 核心满载，而不是三个核心全部满载。

## 结果分析

- 静态类别 FP16 比 Rockchip 原始 FP16 的平均延迟降低约 38.6%，吞吐量提高约 1.63 倍。
- 静态类别 INT8 比静态类别 FP16 的平均延迟降低约 58.4%，吞吐量提高约 2.41 倍。
- 静态类别 INT8 比 Rockchip 原始 FP16 的平均延迟降低约 74.5%，吞吐量提高约 3.92 倍。
- INT8 单路连续推理约 14 FPS，可以跟上 10 Hz 图像输入。
- 静态 FP16 约 5.8 FPS，面对 10 Hz 图像输入时需要丢帧、降采样或者降低播放速率。
- 三个模型都主要使用 Core0。设置 `NPU_CORE_0_1_2` 表示允许 runtime 使用三个核心，但不保证单次推理均匀切分到三个核心。
- 如果目标是提高多帧总体吞吐量，可以测试多个 RKNN context 并行并分别绑定核心；这种方案不一定降低单帧延迟，并会增加内存占用。

## 预处理参考

在第一轮测试中，平均预处理时间为：

| 调用流程 | 平均预处理耗时 |
|---|---:|
| Rockchip 原始模型流程 | 4.38 ms |
| 静态类别模型流程 | 1.03 ms |

第一轮不具备可信的逐核心 NPU 数据，因为普通 devfreq `load` 节点在空闲时也保持 100%。正式负载结果使用已经验证空闲为 0% 的 debugfs 接口重新测得。

## 使用建议

- 优先保证检测精度时使用静态 FP16。
- 优先保证 10 Hz 实时处理时使用静态 INT8，但需要继续扩充室内校准图片，改善框偏移和误检。
- 比较模型效果时应保持完全一致的图片、置信度、NMS IoU、类别顺序和预处理方式。
