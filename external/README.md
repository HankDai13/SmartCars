# External Source

原厂小车源码来自：

```text
https://gitee.com/HUAWEI-ASCEND/ascend-devkit/tree/master/src/E2E-Sample/Car
```

使用脚本 sparse checkout：

```powershell
.\scripts\fetch_ascend_car.ps1
```

或：

```bash
bash scripts/fetch_ascend_car.sh
```

拉取后主要复用：

- `Car/python/src/actions/base_action.py`
- `Car/python/src/actions/complex_actions.py`
- `Car/python/src/utils/controller.py`
- `Car/python/src/models/lfnet.py`
- `Car/python/src/models/yolov5.py`
- `Car/python/src/scenes/helper.py`
- `Car/python/src/scenes/lane_following.py`

本仓库的 ROS2 层应依赖这些模块作为硬件/模型适配层，而不是把课程主逻辑继续写在原 demo 的 scene 里。

