import torch.nn as nn
import torch
from ultrahelper.cfg.yolov8_pose import Default_CFG  # import from correct YAML module


def get_activation(name):
    name = (name or 'silu').lower()
    if name == 'silu':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ModifiedConv(nn.Module):  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        # Decide activation
        if isinstance(act, str):
            self.act = get_activation(act)
        elif isinstance(act, nn.Module):
            self.act = act
        elif act is False:
            self.act = nn.Identity()
        else:
            # Pull default from cfg
            default_act = Default_CFG.get('custom', {}).get('act', 'silu')
            self.act = get_activation(default_act)
        print(f'activation function is: {self.act}')


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))





class ModifiedSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ModifiedConv(c1, c_, 1, 1)
        self.cv2 = ModifiedConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

