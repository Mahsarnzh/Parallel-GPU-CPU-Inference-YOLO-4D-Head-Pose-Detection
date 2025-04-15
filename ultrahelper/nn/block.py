from ultralytics.nn.modules.block import C2f
from .register import register_module


@register_module('base')
@register_module('repeat')
class TracableC2f(C2f):
    """
    Equivalent implementation to C2f but symbolically tracables.
    """

