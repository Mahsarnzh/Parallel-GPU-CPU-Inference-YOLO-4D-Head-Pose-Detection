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



# import torch
# import torch.nn as nn
# from ultralytics.nn.modules.head import Pose
# from .register import register_module


# @register_module('head')
# class ModifiedPose(Pose):
#     """
#     Refactored Pose module for hardware-constrained platforms.
#     `forward_head` contains all convolution operations (4D tensors).
#     `forward_postprocessor` contains all view/reshape/non-4D ops.
#     """

#     def get_head(self) -> 'ModifiedPoseHead':
#         return ModifiedPoseHead(self)

#     def get_postprocessor(self) -> 'ModifiedPosePostprocessor':
#         return ModifiedPosePostprocessor(self)

#     def forward_head(self, x):
#         """All 4D-tensor computations including convolutions & Detect.forward."""
#         return [self.cv4[i](x[i]) for i in range(self.nl)]

#     def forward_postprocessor(self, feats):
#         """
#         Handle postprocessing including reshaping and keypoint decoding.
#         """
#         bs = feats[0].shape[0]
#         kpt = torch.cat([f.view(bs, self.nk, -1) for f in feats], -1)  # (bs, 17*3, h*w)

#         if self.training:
#             x = super(ModifiedPose, self).forward(feats)  # or manually call Detect.forward
#             return x, kpt

#         pred_kpt = self.kpts_decode(bs, kpt)
#         x = super(ModifiedPose, self).forward(feats)
#         return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))


#     def forward(self, x):
#         feats = self.forward_head(x)
#         preds = self.forward_postprocessor(feats)
#         return preds


# class ModifiedPoseHead(ModifiedPose):
#     def __init__(self, mpose: ModifiedPose):
#         nn.Module.__init__(self)
#         self.__dict__.update(mpose.__dict__)

#     def forward(self, x):
#         with torch.no_grad():
#             return self.forward_head(x)


# class ModifiedPosePostprocessor(ModifiedPose):
#     def __init__(self, mpose: ModifiedPose):
#         nn.Module.__init__(self)
#         self.__dict__.update(mpose.__dict__)

#     def forward(self, feats):
#         with torch.no_grad():
#             return self.forward_postprocessor(feats)

