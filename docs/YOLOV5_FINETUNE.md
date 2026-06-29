# YOLOv5 Fine-Tuning Guide

这份文档给负责训练的队友或 Codex 使用。当前路线是：本地/云 GPU 用 Ultralytics YOLOv5 预训练权重微调，导出 ONNX；Atlas 200I DK A2 端用 CANN/ATC 转 OM 后做推理验证。官方小车样例和往届代码都不是在 Atlas 上训练 YOLO，而是训练后把模型放到车端运行。

## 1. 代码和数据位置

关键路径：

```text
data/yolo/                         # 统一后的 YOLO 数据集，构建脚本生成
configs/yolov5_finetune.yaml       # 本项目默认训练配置
ml/yolov5/train_smartcar.py        # 训练 wrapper：校验数据 + 调 YOLOv5 train.py
ml/yolov5/training_status.py       # 查看训练进度和产物路径
models/yolov5/runs/                # 训练输出目录
models/yolov5/smartcar_yolov5.onnx # 导出的 ONNX
```

类别固定为 7 类：

```text
0 left
1 right
2 turnaround
3 park
4 person
5 obstacle
6 crosswalk
```

## 2. 环境配置

推荐在本地 WSL、Linux GPU 服务器或 AutoDL 上训练。Atlas 端只负责推理部署和实车测试。

创建环境：

```bash
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install -U pip setuptools wheel
```

安装 PyTorch：按机器 CUDA 版本选择官方命令。CPU 也能跑通流程但训练会很慢。随后安装项目和 YOLOv5 依赖：

```bash
pip install -r requirements-ml.txt
bash ml/yolov5/fetch_yolov5.sh
pip install -r external/yolov5/requirements.txt
```

如果不想让 YOLOv5 自动连 Weights & Biases，训练 wrapper 默认会设置 `WANDB_MODE=disabled`。

## 3. 构建和检查数据

如果 `data/imported` 或 relabel 结果有更新，先重构建：

```bash
python tools/datasets/build_imported_datasets.py
python tools/datasets/validate_yolo_labels.py --labels data/yolo/labels --num-classes 7
```

数据结构必须是：

```text
data/yolo/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

训练脚本启动前还会再次检查图片和标签是否一一配对、类别 id 是否越界、bbox 是否归一化到 `[0, 1]`。

## 4. 训练配置

默认配置在 `configs/yolov5_finetune.yaml`。第一版建议保持：

```yaml
yolov5:
  weights: yolov5s.pt
train:
  img_size: 640
  batch_size: 16
  epochs: 120
  optimizer: SGD
  patience: 30
  cache: ram
```

选择建议：

- `yolov5s.pt`：默认，速度快，适合第一天先跑通。
- `yolov5n.pt`：更轻，适合快速冒烟测试或车端延迟压力大时尝试。
- `yolov5m.pt`：可能精度更高，但训练和推理更慢，后期再试。
- `batch_size`：显存不够就从 16 降到 8 或 4。
- `epochs`：先用 5 跑冒烟，再用 80-120 跑正式版。

## 5. 启动训练

快速冒烟：

```bash
python ml/yolov5/train_smartcar.py \
  --config configs/yolov5_finetune.yaml \
  --epochs 5 \
  --batch-size 8 \
  --name smoke_yolov5s
```

正式训练：

```bash
python ml/yolov5/train_smartcar.py --config configs/yolov5_finetune.yaml
```

指定 GPU：

```bash
python ml/yolov5/train_smartcar.py --device 0
```

断点续训：

```bash
python ml/yolov5/train_smartcar.py --resume models/yolov5/runs/smartcar_yolov5/weights/last.pt
```

旧 shell 入口仍然可用：

```bash
bash ml/yolov5/train_yolov5.sh
```

训练输出：

```text
models/yolov5/runs/<run_name>/
  weights/best.pt
  weights/last.pt
  results.csv
  results.png
  confusion_matrix.png
  PR_curve.png
```

## 6. 查看训练进度

命令行查看最新 run：

```bash
python ml/yolov5/training_status.py
```

第一次训练启动前没有 run 目录是正常的，这个命令会提示先启动训练。

持续刷新：

```bash
python ml/yolov5/training_status.py --watch 30
```

TensorBoard：

```bash
tensorboard --logdir models/yolov5/runs --host 0.0.0.0 --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

如果是在远程服务器训练，用 SSH 端口转发：

```bash
ssh -L 6006:127.0.0.1:6006 user@server
```

重点看：

- `metrics/mAP_0.5`：主要判断检测能不能用。
- `metrics/precision` / `metrics/recall`：观察误检和漏检。
- `val/box_loss`：框定位是否还在下降。
- `confusion_matrix.png`：看 `park` 和 `obstacle` 是否还混。
- `val_batch*_pred.jpg`：人工看预测框是否合理。

## 7. 导出 ONNX

训练完成后把最优权重导出：

```bash
WEIGHTS=models/yolov5/runs/smartcar_yolov5/weights/best.pt \
OUTPUT=models/yolov5/smartcar_yolov5.onnx \
bash ml/yolov5/export_onnx.sh
```

如果 YOLOv5 因为同名目录已存在自动生成了 `smartcar_yolov52` 之类的新 run，导出时把 `WEIGHTS` 改成实际 run 目录下的 `weights/best.pt`。

导出后建议本地先做一次 ONNXRuntime 冒烟测试，至少确认模型能加载。后续如果 ROS2 推理节点还没补完整，可以先用 YOLOv5 自带 `detect.py` 对若干验证图跑 `.pt`，再用我们自己的 ONNX 节点做对齐。

## 8. 转 OM 和部署关系

官方小车样例的路径是：

```text
external/ascend-devkit/src/E2E-Sample/Car/
```

官方代码结构里 YOLO 是车端推理模块，权重放在 `python/weights`，不是训练入口。我们这里保持：

```text
训练机: data/yolo -> YOLOv5 fine-tune -> best.pt -> ONNX
Atlas: ONNX -> ATC -> OM -> ROS2/官方推理代码加载
```

Atlas 上转换：

```bash
bash tools/export/atc_convert.sh \
  models/yolov5/smartcar_yolov5.onnx \
  models/yolov5/smartcar_yolov5 \
  "images:1,3,640,640" \
  Ascend310B4
```

如果课程镜像要求 `Ascend310B1`，按板子 CANN 环境实际 `soc_version` 修改。生成：

```text
models/yolov5/smartcar_yolov5.om
```

## 9. 推荐执行顺序

第一轮：

```bash
python tools/datasets/build_imported_datasets.py
python tools/datasets/validate_yolo_labels.py --labels data/yolo/labels --num-classes 7
bash ml/yolov5/fetch_yolov5.sh
pip install -r external/yolov5/requirements.txt
python ml/yolov5/train_smartcar.py --epochs 5 --batch-size 8 --name smoke_yolov5s
python ml/yolov5/training_status.py
```

确认没问题后：

```bash
python ml/yolov5/train_smartcar.py --config configs/yolov5_finetune.yaml
WEIGHTS=models/yolov5/runs/smartcar_yolov5/weights/best.pt \
OUTPUT=models/yolov5/smartcar_yolov5.onnx \
bash ml/yolov5/export_onnx.sh
```

## 10. 常见问题

`CUDA out of memory`：把 `batch_size` 降到 8、4 或 2。

`YOLOv5 not found`：先跑 `bash ml/yolov5/fetch_yolov5.sh`。

`images have no label file`：说明另一个进程构建数据时中断了，重新跑 `python tools/datasets/build_imported_datasets.py`。

`park` 和 `obstacle` 混淆严重：优先回看 `reports/dataset_visualization*` 和 `confusion_matrix.png`，必要时继续清洗 park 标注。
