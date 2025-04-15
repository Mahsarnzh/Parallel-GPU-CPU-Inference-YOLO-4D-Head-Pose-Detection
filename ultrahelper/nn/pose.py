import torch
from ultralytics.nn.modules.head import Pose
from .register import register_module
import torch.nn as nn

@register_module('head')
class ModifiedPose(Pose):
    """
    Modify Pose module without changing its architecture. 
    Implement 'forward_head' and 'forward_postprocessor' methods
    """
    def get_head(self,)->'ModifiedPoseHead':
        return ModifiedPoseHead(self)
    def get_postprocessor(self,)->'ModifiedPosePostprocessor':
        return ModifiedPosePostprocessor(self)
    def forward_head(self,x):
        """
        Contains maximum amount of operations excluding 'view' methods 
        """
        return super().forward(x)
    def forward_postprocessor(self,x):
        """
        Contains everything else left (including 'view' methods)
        """
        return x
    def forward(self,x):
        feats = self.forward_head(x)
        preds = self.forward_postprocessor(feats)
        return preds
    
class ModifiedPoseHead(ModifiedPose):
    def __init__(self,mpose:ModifiedPose):
        nn.Module.__init__(self,)
        self.__dict__.update(mpose.__dict__)
    def forward(self,x):
        with torch.no_grad():
            return self.forward_head(x)

class ModifiedPosePostprocessor(ModifiedPose):
    def __init__(self,mpose:ModifiedPose):
        nn.Module.__init__(self,)
        self.__dict__.update(mpose.__dict__)
    def forward(self,feats):
        with torch.no_grad():
            return self.forward_postprocessor(feats)