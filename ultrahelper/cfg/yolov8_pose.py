import yaml
from pathlib import Path
from dataclasses import dataclass, field
import os
import torch.nn as nn


@dataclass
class YoloV8Config:
    path: str = field(default=os.environ.get(
        "YoloV8Config",
        Path(__file__).parent / "yolov8-pose.yaml"
    ))
    c: dict = None

    activations = {c.__class__.__name__.lower(): c for c in (
            nn.SiLU(),
            nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.GELU(),
            nn.Identity()
    )}
    
    @property
    def cfg(self):
        if self.c is None:
            self.read()
        return self.c
    
    def read(self):
        with open(self.path, "r") as f:
            self.c = yaml.safe_load(f)
        return self.c
    
    def get_default_activation(self):
        return self.get_activation(self.cfg["custom"]["act"])
    
    @classmethod
    def get_activation(cls, name: str = None):
        n = (name or 'silu').lower()
        if n not in cls.activations:
            raise ValueError(f"Unsupported activation: {n} ({name})")
        return cls.activations[n]

    
