# Model Artifacts

模型文件通常较大，不提交 git。建议命名保持稳定：

```text
models/yolov5/smartcar_yolov5.pt
models/yolov5/smartcar_yolov5.onnx
models/yolov5/smartcar_yolov5.om

models/lfnet/lfnet.pth
models/lfnet/lfnet.onnx
models/lfnet/lfnet.om

models/lanenet_hnet/lanenet_hnet.pth
models/lanenet_hnet/lanenet_hnet.onnx
models/lanenet_hnet/lanenet_hnet.om
```

现场验收前，把模型版本、训练数据规模、地图/光照覆盖范围、mAP/角度误差/推理 FPS 记录到 `reports/`。

