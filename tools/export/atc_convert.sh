#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <model.onnx> <output_prefix> <input_shape> [soc_version]"
  echo "Example: $0 models/lfnet/lfnet.onnx models/lfnet/lfnet 'input:1,3,180,320' Ascend310B4"
  exit 2
fi

MODEL="$1"
OUTPUT="$2"
INPUT_SHAPE="$3"
SOC_VERSION="${4:-Ascend310B4}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [ -f "$CANN_ENV" ]; then
  # shellcheck source=/dev/null
  source "$CANN_ENV"
fi

atc \
  --model="$MODEL" \
  --framework=5 \
  --output="$OUTPUT" \
  --soc_version="$SOC_VERSION" \
  --input_shape="$INPUT_SHAPE"

