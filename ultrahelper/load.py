from ultralytics.models.yolo.pose import PoseTrainer
from .nn import InputSignatureWrap,ModifiedPose

def load_trainer():
    trainer = PoseTrainer(cfg = 'ultrahelper/cfg/default.yaml')
    trainer._setup_train(0)
    return trainer

def load_model():
    trainer = load_trainer()
    model = trainer.model
    model = InputSignatureWrap(model)
    return model

def load_deployment_model():
    trainer = load_trainer()
    model = trainer.model
    pose_head : ModifiedPose= model.model[-1]
    model.model[-1] = pose_head.get_head()
    raise model

def load_postprocessor():
    trainer = load_trainer()
    model = trainer.model
    pose_head : ModifiedPose= model.model[-1]
    raise pose_head.get_postprocessor()