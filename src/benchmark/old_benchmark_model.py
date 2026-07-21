import os
import sys
import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from src.utils.layer_factory import SequentialBuilder


class BenchmarkModel(nn.Module):
    def __init__(self, SETTINGS):
        super().__init__()


        self.signal_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.signal_layers})

        self.mask_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.mask_layers})
        
        self.fusion_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.end_layers})
        
        # Find signal input-output dimensions
        self.signal_dim = None
        self.signal_out_features = None
        for l in SETTINGS.MODEL.signal_layers:
            if l["type"] == "linear":
                if self.signal_dim ==None: self.signal_dim = l["params"]["in_features"]
                self.signal_out_features = l["params"]["out_features"]
        
        # Find mask input-output dimensions
        self.mask_dim = None
        self.mask_out_features = None
        for l in SETTINGS.MODEL.mask_layers:
            if l["type"] == "linear":
                if self.mask_dim ==None: self.mask_dim = l["params"]["in_features"]
                self.mask_out_features = l["params"]["out_features"]
        
        # Find end_model input-output dimensions
        self.end_model_dim = None
        self.end_model_out_features = None
        for l in SETTINGS.MODEL.end_layers:
            if l["type"] == "linear":
                if self.end_model_dim ==None: self.end_model_dim = l["params"]["in_features"]
                self.end_model_out_features = l["params"]["out_features"]
    
        if  self.mask_dim is None or self.mask_out_features is None:
            raise ValueError("Mask input and output dimensions could not be determined. Make sure the mask_model linear")
        
        if  self.signal_dim is None or self.signal_out_features is None:
            raise ValueError("Signal input and output dimensions could not be determined. Make sure the signal_model linear")
        
        if  self.end_model_dim is None or self.end_model_out_features is None:
            raise ValueError("The end_model input and output dimensions could not be determined. Make sure the end_model_model linear")
        
        if self.end_model_dim != self.signal_out_features + self.mask_out_features :
            raise ValueError(f"The `end_model` input size (currently {self.end_model_dim}) \
                must be equal to the signal+mask output size, (currently {self.signal_out_features + self.mask_out_features})")
        
    def forward(self, x):

        x_signal = x[..., :self.signal_dim] 
        x_mask = x[..., self.signal_dim:self.signal_dim + self.mask_dim]

        h_signal = self.signal_mlp(x_signal)
        h_mask = self.mask_mlp(x_mask)

        h = torch.cat([h_signal, h_mask], dim=-1)
        y = self.fusion_mlp(h)
        return y
