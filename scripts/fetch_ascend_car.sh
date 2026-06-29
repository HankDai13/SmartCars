#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-external/ascend-devkit}"
BRANCH="${BRANCH:-master}"

if [ -d "$DESTINATION" ]; then
  echo "Destination already exists: $DESTINATION"
  echo "Run 'git -C $DESTINATION pull' if you need to update it."
  exit 0
fi

git clone --depth 1 --filter=blob:none --sparse --branch "$BRANCH" \
  https://gitee.com/HUAWEI-ASCEND/ascend-devkit.git "$DESTINATION"

git -C "$DESTINATION" sparse-checkout set src/E2E-Sample/Car

echo "Fetched car source to $DESTINATION/src/E2E-Sample/Car"

