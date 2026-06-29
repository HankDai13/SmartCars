# YOLOv5 Fine-Tuning Pipeline

目标：基于 YOLOv5 预训练权重微调一个统一检测模型，用于识别转向标识、掉头、泊车、行人、障碍、人行道等目标。

主要文档见 [docs/YOLOV5_FINETUNE.md](../../docs/YOLOV5_FINETUNE.md)。

## 1. 获取 YOLOv5 源码

```bash
bash ml/yolov5/fetch_yolov5.sh
```

默认拉到：

```text
external/yolov5
```

## 2. 构建和检查数据

```bash
python tools/datasets/build_imported_datasets.py
python tools/datasets/validate_yolo_labels.py --labels data/yolo/labels --num-classes 7
```

## 3. 微调训练

```bash
python ml/yolov5/train_smartcar.py --config configs/yolov5_finetune.yaml
```

输出权重：

```text
models/yolov5/runs/smartcar_yolov5/weights/best.pt
```

查看进度：

```bash
python ml/yolov5/training_status.py
tensorboard --logdir models/yolov5/runs --host 0.0.0.0 --port 6006
```

## 4. 导出 ONNX / OM

```bash
bash ml/yolov5/export_onnx.sh
bash tools/export/atc_convert.sh \
  models/yolov5/smartcar_yolov5.onnx \
  models/yolov5/smartcar_yolov5 \
  "images:1,3,640,640" \
  Ascend310B4
```
