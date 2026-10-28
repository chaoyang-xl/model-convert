# YOLO-World → RKNN 转换工作区

该目录保存固定 12 个室内类别的 YOLO-World 模型转换、INT8 校准、精度对比和 RK3588 部署材料。

原始模型与转换产物都不入库（见 `.gitignore`），克隆后按下文「原始模型下载」补齐即可复现。

## 前置依赖

已验证的转换环境（x86 Ubuntu，conda 环境 `rknn_convert`）：

| 组件 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.14.0+cu130 |
| ultralytics | 8.4.158（导出用 `opset=12`） |
| onnx / onnxruntime | 1.16.1 / 1.23.2 |
| rknn-toolkit2 | 2.3.2 |

板端（Orange Pi 5 Plus / RK3588）：RKNN Runtime **2.3.2**、RKNPU Driver **0.9.6**，见 `reports/RKNN_200_FRAME_BENCHMARK.md`。

```bash
pip install ultralytics onnx onnxruntime
pip install rknn-toolkit2==2.3.2   # 仅 x86 转换机需要，板端不用
```

版本对齐很重要：`rknn-toolkit2` 与板端 Runtime 大版本不一致时，模型可能加载失败。

## 原始模型下载

### 1. YOLO-World v2 (s)

- 官方地址：<https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt>
- 放置位置：`models/source/yolov8s-worldv2.pt`

```bash
mkdir -p models/source
wget -O models/source/yolov8s-worldv2.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
```

也可以不预下载，首次调用 `model.set_classes()` 时 ultralytics 会自动拉取；但离线或网络不稳时容易卡住，建议先手动下好。

### 2. CLIP ViT-B/32 文本编码器

`set_classes()` 依赖 OpenAI CLIP 的 `ViT-B-32.pt`，用于把类别名编码成 YOLO-World 的文本向量。ultralytics 会按 `clip/ViT-B-32.pt` 的相对路径在权重目录下查找，本项目两个位置都会用到：

- `models/dependencies/clip/ViT-B-32.pt`
- `weights/clip/ViT-B-32.pt`

- 官方地址：<https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt>

```bash
mkdir -p models/dependencies/clip weights/clip
wget -O models/dependencies/clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
cp models/dependencies/clip/ViT-B-32.pt weights/clip/ViT-B-32.pt
```

若该域名访问不了，任何来源的 CLIP ViT-B/32 权重都可以，文件名必须保持 `ViT-B-32.pt`。

### 3. 校准数据（不是模型权重）

INT8 量化需要 100 张室内连续帧（`frame*.jpg`，640×640 输入）。图片和 `dataset.txt` 都不入库：

- 图片放 `data/calibration/images/`；
- `dataset.txt` 每行一个路径，**必须是绝对路径**（rknn-toolkit2 按进程 cwd 解析相对路径，脚本可任意目录启动），所以由本机生成：

```bash
mkdir -p data/calibration/images          # 放入 100 张 jpg
find "$PWD/data/calibration/images" -name "*.jpg" | sort > data/calibration/dataset.txt
```

清单不入库也是为了不把本机路径带进公开仓库。想换数据集直接替换 jpg 并重新执行上面第二条命令。

## 目录结构

```text
models/
  source/          原始 YOLO-World PT
  fixed_pt/        已写入室内类别的 PT
  onnx/            正式 ONNX 与拆分输出 ONNX
  rknn/            FP16、INT8 RKNN 模型
  intermediate/    ONNX 优化过程中的中间模型
  dependencies/    CLIP 等模型依赖
weights/
  clip/            CLIP 权重的另一份副本（部分脚本的默认查找位置）
scripts/
  conversion/      PT → ONNX → RKNN 转换脚本
  evaluation/      PT、ONNX、RKNN 对比和单图检测
  deployment/      ROS bag/图像话题部署脚本
data/
  calibration/     INT8 校准图片及 dataset.txt
  samples/         单图测试输入
results/            检测图片和历史对比结果
reports/            性能测试报告
artifacts/          可再生成的缓存或临时产物
```

所有本地转换和评测脚本均以本文件所在目录作为项目根目录定位资源，因此可从任意工作目录启动。

## 转换顺序

```bash
python3 scripts/conversion/export_worldv2.py
python3 scripts/conversion/split_onnx_output.py
python3 scripts/conversion/convert_rknn_fp.py
python3 scripts/conversion/convert_rknn_int8.py
```

`export_worldv2.py` 将 12 个室内类别写入模型并导出静态 ONNX；`split_onnx_output.py` 把检测框和类别分数拆成两个输出，供当前 INT8 流程使用。

## 常用评测

```bash
python3 scripts/evaluation/compare_pt_onnx.py
python3 scripts/evaluation/compare_onnx_rknn.py
python3 scripts/evaluation/compare_onnx_rknn_int8.py
python3 scripts/evaluation/detect_int8.py
```

板端 200 帧延迟与 NPU 负载结果见：

```text
reports/RKNN_200_FRAME_BENCHMARK.md
```

## 当前固定类别

`chair, sofa, table, cabinet, shelf, monitor, computer, television, plant, trash can, door, desk`

仓库里另有一组 `*_home_indoor*` 变体，使用 37 类，类别表见 `scripts/conversion/home_indoor_classes.txt`；两套模型的 ONNX/RKNN 不能混用。

类别数量或顺序变化后，需要从固定类别 PT/ONNX 开始重新导出，随后重新进行 INT8 校准。
