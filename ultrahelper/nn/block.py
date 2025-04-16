from ultralytics.nn.modules.block import C2f
from .register import register_module
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck
import torch.nn as nn
import torch

@register_module('base')
@register_module('repeat')
class TracableC2f(nn.Module):
    """
    Equivalent implementation to C2f but symbolically traceable by torch.fx.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1, y2 = self.cv1(x).split((self.c, self.c), 1)
        outputs = [y1, y2]
        for b in self.bottlenecks:
            y2 = b(y2)
            outputs.append(y2)
        return self.cv2(torch.cat(outputs, dim=1))

