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

## 两条转换路线

两条路线**共用第一段**：把 12 个室内类别写进模型，导出静态 ONNX。

```bash
python3 scripts/conversion/export_worldv2.py
# -> models/onnx/yolov8s-worldv2_indoor.onnx   单输出 [1, 16, 8400]（4 框参数 + 12 类）
```

导出参数固定为 `imgsz=640, batch=1, dynamic=False, simplify=False, opset=12, half=False`。

### 路线 A：单输出 + FP16（不量化）

适合先跑通、看效果，**不需要校准集**。

```bash
python3 scripts/conversion/convert_rknn_fp.py
```

| 文件 | 说明 |
|---|---|
| 输入 `models/onnx/yolov8s-worldv2_indoor.onnx` | 单输出 `output0 [1,16,8400]` |
| 输出 `models/rknn/yolov8s-worldv2_indoor_fp.rknn` | FP16（`do_quantization=False`），单输出 |

后处理：`pred[:, 0:4]` 是框、`pred[:, 4:]` 是类别分数，见 `scripts/evaluation/compare_onnx_rknn.py`。

### 路线 B：拆分双输出 + INT8 量化

INT8 必须先拆输出，原因见下。

```bash
python3 scripts/conversion/split_onnx_output.py   # 摘两个中间张量当输出
python3 scripts/conversion/convert_rknn_int8.py   # 需要 data/calibration/dataset.txt
```

| 文件 | 说明 |
|---|---|
| 输入 `models/onnx/yolov8s-worldv2_indoor.onnx` | 单输出 |
| 中间 `models/onnx/yolov8s-worldv2_indoor_split.onnx` | **仍是 1 个文件**，但含 **2 个输出张量**：`/model.22/Mul_2_output_0` 框 `[1,4,8400]`、`/model.22/Sigmoid_output_0` 分数 `[1,12,8400]` |
| 输出 `models/rknn/yolov8s-worldv2_indoor_split_int8.rknn` | INT8，**单文件**，推理返回长度 2 的 list |

后处理：`outputs[0]` 是框、`outputs[1]` 是分数，**分数已过 sigmoid，不要再 sigmoid**，见 `scripts/evaluation/detect_int8.py`。

拆分节点写死在脚本里：

```text
BOX_TENSOR = "/model.22/Mul_2_output_0"
CLS_TENSOR = "/model.22/Sigmoid_output_0"
```

### 为什么必须拆

INT8 量化时每个张量只有一组 scale。单输出把框坐标（量级 0~640）和类别分数（0~1）拼在同一维，一组 scale 必然顾此失彼：要么框精度掉，要么 12 类分数被压成几个灰阶。拆开后两条分支各自量化，这是 INT8 能做到 71 ms 且精度可接受的前提。

### 两条路线对照

| | 路线 A | 路线 B |
|---|---|---|
| 输出张量 | 1 个（框 + 分数合并） | 2 个（框、分数分开） |
| 量化 | 不量化（FP16） | INT8 |
| 需要校准集 | 不需要 | 需要 `data/calibration/dataset.txt` |
| 生成脚本 | `convert_rknn_fp.py` | `split_onnx_output.py` + `convert_rknn_int8.py` |
| 产物 | `_indoor_fp.rknn` | `_indoor_split_int8.rknn` |
| 对比脚本 | `compare_onnx_rknn.py` | `compare_onnx_rknn_int8.py` |

> 注意：`models/rknn/yolov8s-worldv2_indoor_int8.rknn`（单输出 INT8）是旧版本遗留，当前脚本不再生成它，不要与 `_split_int8` 混用。

## 一键版：36 类 home 变体（FP16 单输出）

`scripts/conversion/convert_home_indoor_fp16.py` 把路线 A 的两步合成**一条命令**，类别从 `home_indoor_classes.txt` 读取（36 类，输出 `[1,40,8400]`）：

```bash
python3 scripts/conversion/convert_home_indoor_fp16.py
```

它除了等价于 `export_worldv2.py` + `convert_rknn_fp.py`，还额外做了：类别空值/重复检查、保存后重载比对 `names`、`onnx.checker`、输出形状断言 `[1,4+C,8400]`、每个 rknn 调用返回值检查、`finally: rknn.release()`。

产物是 `models/rknn/yolov8s-worldv2_home_indoor_fp16.rknn`。它**只做单输出 FP16**，不涉及拆分与 INT8；12 类 indoor 那条仍需上面的两步（类别写死在 `export_worldv2.py` 里，改类要改代码）。

## 常用评测

| 脚本 | 对应路线 | 作用 |
|---|---|---|
| `compare_pt_onnx.py` | 单输出 | PT 与 ONNX 的框/分数对比 |
| `compare_onnx_rknn.py` | 路线 A | ONNX 与 FP16 RKNN 对比 |
| `compare_onnx_rknn_int8.py` | 路线 B | split ONNX 与 INT8 RKNN 对比 |
| `detect_int8.py` | 路线 B | 单图检测并出图 |
| `inspect_rknn.py` | — | 查看模型输入输出张量 |

板端 200 帧延迟与 NPU 负载结果见 `reports/RKNN_200_FRAME_BENCHMARK.md`。

## 当前固定类别

`chair, sofa, table, cabinet, shelf, monitor, computer, television, plant, trash can, door, desk`

仓库里另有一组 `*_home_indoor*` 变体，使用 36 类，类别表见 `scripts/conversion/home_indoor_classes.txt`；两套模型的 ONNX/RKNN 不能混用。

类别数量或顺序变化后，需要从固定类别 PT/ONNX 开始重新导出，随后重新进行 INT8 校准。
