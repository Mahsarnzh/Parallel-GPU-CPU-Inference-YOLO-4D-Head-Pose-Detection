import torch.nn as nn
import torch
from ultrahelper.cfg.yolov8_pose import YoloV8Config  # import from correct YAML module
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SPPF

cfg = YoloV8Config()


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


default_act = cfg.get_default_activation()


class ModifiedConv(Conv): 
    default_act = YoloV8Config().get_default_activation()  # .get_activation()  # default activation
 
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    print(f'activation function is: {default_act}')







class ModifiedSPPF(SPPF):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        c1 = args[0] if len(args) > 0 else kwargs.get("c1")
        c2 = args[1] if len(args) > 1 else kwargs.get("c2")
        c_ = c1 // 2

        self.cv1 = ModifiedConv(c1, c_, 1, 1)
        self.cv2 =  ModifiedConv(c_ * 4, c2, 1, 1)

    # def __init__(self, c1, c2, k=5):
    #     """
    #     Initialize the SPPF layer with given input/output channels and kernel size.

    #     Args:
    #         c1 (int): Input channels.
    #         c2 (int): Output channels.
    #         k (int): Kernel size.

    #     Notes:
    #         This module is equivalent to SPP(k=(5, 9, 13)).
    #     """
    #     super().__init__()
    #     c_ = c1 // 2  # hidden channels
    #     self.cv1 = ModifiedConv(c1, c_, 1, 1)
    #     self.cv2 = ModifiedConv(c_ * 4, c2, 1, 1)
    #     self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    # def forward(self, x):
    #     """Apply sequential pooling operations to input and return concatenated feature maps."""
    #     y = [self.cv1(x)]
    #     y.extend(self.m(y[-1]) for _ in range(3))
    #     return self.cv2(torch.cat(y, 1))


