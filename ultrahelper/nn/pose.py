import torch
from ultralytics.nn.modules.head import Pose, Detect
from .register import register_module
import torch.nn as nn
from ..utils.monitor_4d import monitor_4d_ops, register_4d_monitor


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
        bs = x[0].shape[0]
        feats = [self.cv4[i](x[i]) for i in range(self.nl)]  # Pure convolution features (4D tensors)
        det_out = Detect.forward(self, x)  # detection output
        return feats, det_out, bs
    
    def forward_postprocessor(self,x):
        """
        Contains everything else left (including 'view' methods)
        """
        feats, det_out, bs = x
        # build a list of reshaped feature maps
        views = []
        for f in feats:
            views.append(f.view(bs, self.nk, -1))
        kpt = torch.cat(views, dim=-1)
        if self.training:
            return det_out, kpt

        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([det_out, pred_kpt], 1) if self.export else (torch.cat([det_out[0], pred_kpt], 1), (det_out[1], kpt))

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
        with monitor_4d_ops(self):          
            with torch.no_grad():
                return self.forward_postprocessor(feats)