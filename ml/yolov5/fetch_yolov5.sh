#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-external/yolov5}"
BRANCH="${YOLOV5_BRANCH:-v7.0}"

if [ -d "$DESTINATION" ]; then
  echo "Destination already exists: $DESTINATION"
  exit 0
fi

git clone --depth 1 --branch "$BRANCH" https://github.com/ultralytics/yolov5.git "$DESTINATION"

