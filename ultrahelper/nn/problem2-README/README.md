# Extend Activations

This README describes the changes made to support configurable activation functions for the `Conv` and `SPPF` modules in the custom Ultrahelper fork.

## Motivation
By default, Ultralytics’ `Conv` and `SPPF` layers use the SiLU activation. To allow easy switching between SiLU, ReLU, and other activations via configuration, we introduced:

1. **`YoloV8Config`** – a dataclass to load and parse `yolov8-pose.yaml` and expose selected activation.
2. **`ModifiedConv`** – overrides `Conv.default_act` to use the config’s chosen activation.
3. **`ModifiedSPPF`** – replaces internal `Conv` layers with `ModifiedConv` so pooling blocks also respect the config.

## File Changes

### `ultrahelper/cfg/yolov8_pose.py`
- Added `YoloV8Config` class:
  - Reads the YAML config file.
  - Exposes a mapping from activation names (`silu`, `relu`, `gelu`, etc.) to `nn.Module` instances.
  - Implements `get_default_activation()` and `get_activation(name)` methods.

### `ultrahelper/nn/modules/conv.py`
- Imported `YoloV8Config`.
- Defined `ModifiedConv(Conv)`:
  ```python
  class ModifiedConv(Conv):
      default_act = YoloV8Config().get_default_activation()
  ```

### `ultrahelper/nn/modules/block.py` (SPPF update)
- Updated `SPPF` wrapper to use `ModifiedConv` internally:
  ```python
  class ModifiedSPPF(SPPF):
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          c1 = args[0] if len(args) > 0 else kwargs.get("c1")
          c2 = args[1] if len(args) > 1 else kwargs.get("c2")
          c_ = c1 // 2
          self.cv1 = ModifiedConv(c1, c_, 1, 1)
          self.cv2 = ModifiedConv(c_ * 4, c2, 1, 1)
  ```

## Configuration
In `yolov8-pose.yaml`, under a new `custom` section, add an `act` entry:

```yaml
custom:
  act: relu   # options: silu, relu, gelu, identity, leakyrelu
```

This value drives the default activation across all `Conv` and `SPPF` layers.

## Usage
1. Install and import the Ultrahelper package.
2. Edit `ultrahelper/cfg/yolov8-pose.yaml`, set `custom.act`.
3. Run your training or inference as usual:

```bash
python -m ultrahelper --train
# The chosen activation will be applied in the network.
```

## Future Extensions
- Expose per-layer activation overrides via the config.
- Add support for advanced activations (e.g. Swish, Mish).

---
*Authored by Mahsa Raeisinezhad.*

