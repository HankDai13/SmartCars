# LFNet Pipeline

LFNet is the lightweight lane-following baseline. It regresses a steering angle in `[0, 135]`, where `90` means centered.

## Dataset

```text
data/lfnet/images/
data/lfnet/labels.csv
```

CSV format:

```csv
image,angle
000001.jpg,90.0
000002.jpg,82.0
```

## Train

```bash
python ml/lfnet/train.py --config configs/lfnet.example.yaml
```

## Export

```bash
python ml/lfnet/export_onnx.py \
  --checkpoint models/lfnet/runs/baseline/best.pth \
  --output models/lfnet/lfnet.onnx
```

