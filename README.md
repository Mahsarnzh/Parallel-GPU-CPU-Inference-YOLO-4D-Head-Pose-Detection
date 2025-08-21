## ultrahelper

This repository provides a partially implemented package called `ultrahelper`, designed to extend and customize the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework **without modifying its source code**. The goal is to override and extend certain modules while still leveraging the flexibility of Ultralytics’ configuration system.

Custom modules can be defined and referenced through the configuration file:  
`ultrahelper/cfg/yolov8-pose.yaml`.

The infrastructure for this mechanism is already implemented in `ultrahelper` and demonstrated across multiple modules.
## Implemented Tasks

### 1. Resolved Symbolic Tracing Issue in YOLO Model

* Identified and debugged a symbolic tracing error in the YOLOv8 model using `torch.fx`.
* Traced the root cause to runtime-dependent logic within the `C2f` module from `ultralytics.nn.modules.block`.
* Implemented a traceable version of the module (`ModifiedC2f`) in `ultrahelper.nn.block`, ensuring compatibility with PyTorch's symbolic tracer.

### 2. Added Configurable Activation Functions

* Enhanced the model’s flexibility by modifying the `Conv` and `SPPF` modules to support configurable activation functions (e.g., SiLU, ReLU).
* Extended the model's YAML config (`ultrahelper/cfg/yolov8-pose.yaml`) to support activation selection without altering Ultralytics' core code.

### 3. Modularized `ModifiedPose` for Deployment

* Refactored the `ModifiedPose` class in `ultrahelper.nn.pose` to separate hardware-incompatible operations.
* Created two deployable components:

  * `ModifiedPoseHead`: optimized for hardware execution, retaining all convolutional layers.
  * `ModifiedPosePostprocessor`: runs on CPU and handles tensor reshaping and unsupported operations.
* Ensured compliance with hardware constraints (e.g., only 4D tensor operations on device).

### 4. Built Parallel GPU-CPU Inference Pipeline

* Developed a real-time parallel inference pipeline with two decoupled components:

  * A hardware model executing on the GPU.
  * A postprocessing module running on the CPU.
* Utilized `load_hardware_model()` and `load_postprocessor()` from `ultrahelper.load`.
* Implemented real-time performance monitoring, displaying FPS and inference latency while processing video frames continuously.

---

### Setup, after cloning this project

1. Install the `ultralytics` package.

```bash
pip install ultralytics
```

2. Run the following to download the COCO8 dataset and ensure the training pipeline is functional:

```bash
python -m ultrahelper --train
python -m ultrahelper --pipeline
python -m ultrahelper --trace
```

For parallel processing inference and FPS and CPU and GPU time, run:
```bash
python -m ultrahelper --pipeline
```

3. Make sure you have Pytorch version above 2.0 in order to use symbolic tracing.

