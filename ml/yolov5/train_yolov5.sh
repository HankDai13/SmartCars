#!/usr/bin/env bash
set -euo pipefail

YOLOV5_DIR="${YOLOV5_DIR:-external/yolov5}"
DATA_YAML="${DATA_YAML:-configs/yolov5_data.example.yaml}"
WEIGHTS="${WEIGHTS:-yolov5s.pt}"
IMG_SIZE="${IMG_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-80}"
PROJECT="${PROJECT:-models/yolov5/runs}"
NAME="${NAME:-smartcar_yolov5}"

if [ ! -f "$YOLOV5_DIR/train.py" ]; then
  echo "YOLOv5 not found. Run: bash ml/yolov5/fetch_yolov5.sh"
  exit 2
fi

python "$YOLOV5_DIR/train.py" \
  --img "$IMG_SIZE" \
  --batch "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --data "$DATA_YAML" \
  --weights "$WEIGHTS" \
  --project "$PROJECT" \
  --name "$NAME" \
  --cache

