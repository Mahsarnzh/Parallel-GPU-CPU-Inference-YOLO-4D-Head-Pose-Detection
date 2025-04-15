from torch.nn import Module
import torch

class InputSignatureWrap(Module):
    '''
    For models with "args" or "kwargs" expression in forward function, tracing is not possible.
    This module signifies to the tracer, the intended input for the forward function
    '''
    def __init__(self,wrapped_module:Module):
        super().__init__()
        self.wrapped_module = wrapped_module
    def forward(self,x:torch.Tensor):
        return self.wrapped_module(x)