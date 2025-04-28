## ultrahelper

This repository provides a partially implemented package called `ultrahelper`, designed to extend and customize the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework **without modifying its source code**. The goal is to override and extend certain modules while still leveraging the flexibility of Ultralytics’ configuration system.

Custom modules can be defined and referenced through the configuration file:  
`ultrahelper/cfg/yolov8-pose.yaml`.

The infrastructure for this mechanism is already implemented in `ultrahelper` and demonstrated across multiple modules.

---

### Setup

1. Install the `ultralytics` package.

```bash
pip install ultralytics
```
2. Run the following to download the COCO8 dataset and ensure the training pipeline is functional:

```bash
python -m ultrahelper --train
```

3. Make sure you have Pytorch version above 2.0 in order to use symbolic tracing. 

