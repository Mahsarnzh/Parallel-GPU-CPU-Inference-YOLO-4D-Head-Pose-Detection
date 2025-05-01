## ultrahelper

This repository provides a partially implemented package called `ultrahelper`, designed to extend and customize the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework **without modifying its source code**. The goal is to override and extend certain modules while still leveraging the flexibility of Ultralytics’ configuration system.

Custom modules can be defined and referenced through the configuration file:  
`ultrahelper/cfg/yolov8-pose.yaml`.

The infrastructure for this mechanism is already implemented in `ultrahelper` and demonstrated across multiple modules.

#### 4. Implemented a parallel inference pipeline
Built a parallel inference pipeline consisting of two components:
- The **hardware model**, running on a GPU
- The **postprocessor**, running on the CPU

The functions to load the two parts of the model are defined in `ultrahelper.load`:
- `load_hardware_model()`
- `load_postprocessor()`

Your pipeline should:
- Run both components in parallel
- Real-time collect and display while running the pipeline in an infinite loop:
  - Frame rate (FPS)
  - Inference latency

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

