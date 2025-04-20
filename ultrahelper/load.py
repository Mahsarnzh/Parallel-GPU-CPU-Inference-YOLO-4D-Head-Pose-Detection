from ultralytics.models.yolo.pose import PoseTrainer
from .nn import InputSignatureWrap,ModifiedPose
from ultralytics import YOLO
import os
import sys

current_dir = os.path.dirname(__file__)
ultrahelper_root = os.path.abspath(os.path.join(current_dir, '..'))
if ultrahelper_root not in sys.path:
    sys.path.insert(0, ultrahelper_root)

# Resolve model path
def get_best_model_path():
    best_model_path = os.path.join(current_dir, '..', 'runs', 'pose', 'train2', 'weights', 'best.pt')
    best_model_path = os.path.abspath(best_model_path)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"best.pt not found at {best_model_path}")
    return best_model_path

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
    pose_head: ModifiedPose = model.model[-1]
    model.model[-1] = pose_head.get_head()
    return model 


def load_postprocessor():
    trainer = load_trainer()
    model = trainer.model
    pose_head: ModifiedPose = model.model[-1]
    post = pose_head.get_postprocessor()  
    yolo_model = YOLO("yolo11n-pose.pt")
    # yolo_model = YOLO(str(get_best_model_path()))
    post.tracker = yolo_model  # Assign the entire model to `tracker`
    return post


