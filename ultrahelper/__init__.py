# MONKEY PATCHING parse_model function
from .parse import parse_model_extended
import ultralytics.nn.tasks 
ultralytics.nn.tasks.parse_model = parse_model_extended
