import torch
from ultralytics.nn.modules.head import Pose, Detect
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

    def forward_head(self,x):
        """
        Contains maximum amount of operations excluding 'view' methods 
        """
        assert isinstance(x, (list, tuple)), f"Expected input to be list or tuple of tensors, got {type(x)}"
        for i, tensor in enumerate(x):
            assert isinstance(tensor, torch.Tensor), f"Element at index {i} is not a tensor: {type(tensor)}"
            assert tensor.dim() == 4, f"Tensor at index {i} must be 4D (got shape: {tensor.shape})"
        
        bs = x[0].shape[0]
        feats = [self.cv4[i](x[i]) for i in range(self.nl)]  # Pure convolution features (4D tensors)
        print(f"dtype of feats: {type(feats)}")
        if isinstance(feats, (list, tuple)):
            for i, feat in enumerate(feats):
                print("the shape of feat is:", feat.shape)

        det_out = Detect.forward(self, x)  # detection output
        print(type(det_out))
        if isinstance(det_out, torch.Tensor):
            print("det_out is a tensor:", det_out.shape)
        elif isinstance(det_out, (list, tuple)):
            print("x is a list or tuple of length:", len(x))
            for i, val in enumerate(det_out):
                if isinstance(val, list):
                    print(f"det_out[{i}] is a list of length: {len(val)}")
                elif isinstance(val, tuple):
                    print(f"det_out[{i}] is a tuple of length: {len(val)}")
                else:
                    print(f"det_out[{i}] shape: {val.shape}")
        else:
            print("det_out is something else:", type(det_out))


        return feats, det_out, bs
    

    def get_postprocessor(self,)->'ModifiedPosePostprocessor':
        return ModifiedPosePostprocessor(self)
    
    def forward_postprocessor(self,x):
        """
        Contains everything else left (including 'view' methods)
        """
        feats, det_out, bs = x
        kpt = torch.cat([f.view(bs, self.nk, -1) for f in feats], -1)
        assert kpt.dim() != 4, f"dimension of kpt is expected to be different than 4, but got {kpt.dim()}"
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
        with torch.no_grad():
            return self.forward_postprocessor(feats)