# YOLOv5 Pipeline

目标：训练一个统一检测模型，用于识别转向标识、掉头、泊车、行人、障碍、人行道等目标。

## 1. 获取 YOLOv5

```bash
bash ml/yolov5/fetch_yolov5.sh
```

默认拉到：

```text
external/yolov5
```

## 2. 准备数据

```bash
python tools/datasets/validate_yolo_labels.py --labels data/raw/yolo/labels --num-classes 7
python tools/datasets/split_yolo_dataset.py \
  --images data/raw/yolo/images \
  --labels data/raw/yolo/labels \
  --output data/yolo \
  --copy
```

## 3. 训练

```bash
bash ml/yolov5/train_yolov5.sh
```

输出权重：

```text
models/yolov5/runs/smartcar_yolov5/weights/best.pt
```

## 4. 导出 ONNX

```bash
bash ml/yolov5/export_onnx.sh
```

## 5. 转 OM

```bash
bash tools/export/atc_convert.sh \
  models/yolov5/smartcar_yolov5.onnx \
  models/yolov5/smartcar_yolov5 \
  "images:1,3,640,640" \
  Ascend310B4
```

