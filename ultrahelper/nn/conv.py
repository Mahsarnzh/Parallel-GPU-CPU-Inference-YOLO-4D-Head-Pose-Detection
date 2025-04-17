from ultrahelper.cfg.yolov8_pose import YoloV8Config  
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SPPF

cfg = YoloV8Config()

class ModifiedConv(Conv): 
    default_act = YoloV8Config().get_default_activation()  # .get_activation() 
    # print(f'activation function is: {default_act}')


class ModifiedSPPF(SPPF):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        c1 = args[0] if len(args) > 0 else kwargs.get("c1")
        c2 = args[1] if len(args) > 1 else kwargs.get("c2")
        c_ = c1 // 2
        self.cv1 = ModifiedConv(c1, c_, 1, 1)
        self.cv2 =  ModifiedConv(c_ * 4, c2, 1, 1)
