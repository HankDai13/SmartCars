# Training Pipelines

## YOLOv5

用途：路标、行人、障碍物、人行道等目标检测。

数据格式：

```text
data/yolo/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

准备数据：

```bash
python tools/datasets/split_yolo_dataset.py \
  --images data/raw/yolo/images \
  --labels data/raw/yolo/labels \
  --output data/yolo \
  --train 0.8 --val 0.1 --test 0.1
```

训练：

```bash
bash ml/yolov5/train_yolov5.sh
```

导出 ONNX：

```bash
bash ml/yolov5/export_onnx.sh
```

转 OM：

```bash
bash tools/export/atc_convert.sh \
  models/yolov5/smartcar_yolov5.onnx \
  models/yolov5/smartcar_yolov5 \
  "images:1,3,640,640" \
  Ascend310B4
```

## LFNet

用途：轻量端到端巡线角度回归。输入图像，输出转向角 `[0, 135]`，`90` 为居中。

数据格式：

```text
data/lfnet/images/
data/lfnet/labels.csv
```

`labels.csv`：

```csv
image,angle
000001.jpg,90.0
000002.jpg,83.5
```

训练：

```bash
python ml/lfnet/train.py --config configs/lfnet.example.yaml
```

导出：

```bash
python ml/lfnet/export_onnx.py \
  --checkpoint models/lfnet/runs/baseline/best.pth \
  --output models/lfnet/lfnet.onnx
```

转 OM：

```bash
bash tools/export/atc_convert.sh \
  models/lfnet/lfnet.onnx \
  models/lfnet/lfnet \
  "input:1,3,180,320" \
  Ascend310B4
```

## LaneNet + H-Net

用途：更强的车道线实例分割、聚类和拟合，适合复杂地图与多光照泛化。

数据格式：

```text
data/lanenet_hnet/images/
data/lanenet_hnet/binary_masks/
data/lanenet_hnet/instance_masks/
data/lanenet_hnet/splits.csv
```

训练：

```bash
python ml/lanenet_hnet/train.py --config configs/lanenet_hnet.example.yaml
```

导出：

```bash
python ml/lanenet_hnet/export_onnx.py \
  --checkpoint models/lanenet_hnet/runs/baseline/best.pth \
  --output models/lanenet_hnet/lanenet_hnet.onnx
```

转 OM：

```bash
bash tools/export/atc_convert.sh \
  models/lanenet_hnet/lanenet_hnet.onnx \
  models/lanenet_hnet/lanenet_hnet \
  "input:1,3,256,512" \
  Ascend310B4
```

