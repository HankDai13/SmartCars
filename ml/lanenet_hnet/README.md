# LaneNet + H-Net Pipeline

This pipeline is the stronger lane perception route for complex maps and lighting. It predicts:

- binary lane segmentation
- instance embeddings for lane separation
- H-Net affine parameters for perspective normalization experiments

## Dataset

```text
data/lanenet_hnet/images/
data/lanenet_hnet/binary_masks/
data/lanenet_hnet/instance_masks/
data/lanenet_hnet/splits.csv
```

`splits.csv`:

```csv
image,split
000001.jpg,train
000002.jpg,val
```

Masks must share the image stem. Binary masks use 0/255. Instance masks use 0 for background and positive integer ids for lanes.

## Train

```bash
python ml/lanenet_hnet/train.py --config configs/lanenet_hnet.example.yaml
```

## Export

```bash
python ml/lanenet_hnet/export_onnx.py \
  --checkpoint models/lanenet_hnet/runs/baseline/best.pth \
  --output models/lanenet_hnet/lanenet_hnet.onnx
```

