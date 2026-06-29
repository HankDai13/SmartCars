# Data Layout

训练数据不进入 git。建议按任务拆分并用日期/地图/光照做元数据记录。

```text
data/raw/                  原始采集视频和图片
data/processed/            清洗后的中间结果
data/yolo/
  images/train|val|test
  labels/train|val|test
data/lfnet/
  images/
  labels.csv               image,angle
data/lanenet_hnet/
  images/
  binary_masks/
  instance_masks/
  splits.csv               image,split
```

YOLO 标签格式：

```text
class_id x_center y_center width height
```

所有坐标归一化到 `[0, 1]`。
