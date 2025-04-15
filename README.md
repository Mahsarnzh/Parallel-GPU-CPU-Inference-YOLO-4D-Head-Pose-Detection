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

---

### Tasks

#### 1. Trace the model and identify a symbolic tracing issue

Run the following command:

```bash
python -m ultrahelper --trace
```

You’ll encounter an error related to `torch.fx`. PyTorch symbolic tracing works by intercepting operations during model execution to construct a graph. It supports common operations (e.g., convolutions, addition, concatenation) as well as basic data structures (e.g., lists, tuples, dicts).

However, issues arise when the model's behavior depends on **runtime values**. For example, the condition `if tensor.mean() < 0` cannot be evaluated during tracing, since the value of `tensor.mean()` isn't known at trace time.

Follow the error message. You'll see it originates from the `C2f` module, defined in `ultralytics.nn.modules.block`.

Your task:
- Read the implementation of `C2f`
- Implement an equivalent but tracable `forward()` method in the `ModifiedC2f` class, found in `ultrahelper.nn.block`

#### 2. Extend Activations

In the current model configuration (`ultrahelper/cfg/yolov8-pose.yaml`), the `Conv` and `SPPF` modules use the **SiLU** activation function by default.

Your task is to **add flexibility** to these modules so that the activation function can be selected through the config file — for example, allowing a switch between **SiLU** and **ReLU** without modifying Ultralytics’ core code.

#### 3. Split `ModifiedPose` into deployable components

Review the `ModifiedPose` class in `ultrahelper.nn.pose`.

While it's functional for training, we need to restructure it for deployment to accommodate hardware limitations. Specifically, we want to separate it into:

- `ModifiedPoseHead`: Executed on the hardware
- `ModifiedPosePostprocessor`: Executed on the CPU

This is because some operations in `ModifiedPose` are not supported on the target hardware.

Your task:
- Investigate the parent class `ultralytics.nn.modules.head.Pose` to understand what operations it performs
- Refactor the module according to the following criteria:
  1. Keep as many **convolution operations** as possible in `ModifiedPoseHead`, since they run faster on the hardware
  2. The hardware supports only **4D tensors**; any operation that requires a different tensor shape should be moved to `ModifiedPosePostprocessor`


#### 4. Implement a parallel inference pipeline

Build a parallel inference pipeline consisting of two components:

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