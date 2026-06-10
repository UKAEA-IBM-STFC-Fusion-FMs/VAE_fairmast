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
    def __init__(self, SETTINGS, out_dim):
        super().__init__()


        self.signal_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.signal_layers})

        self.mask_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.mask_layers})
        
        self.signal_dim = None
        self.signal_out_features = None
        for l in SETTINGS.MODEL.signal_layers:
            if l["type"] == "linear":
                if self.signal_dim ==None: self.signal_dim = l["params"]["in_features"]
                self.signal_out_features = l["params"]["out_features"]
        
        self.mask_dim = None
        self.mask_out_features = None
        for l in SETTINGS.MODEL.mask_layers:
            if l["type"] == "linear":
                if self.mask_dim ==None: self.mask_dim = l["params"]["in_features"]
                self.mask_out_features = l["params"]["out_features"]
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.signal_out_features + self.mask_out_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):

        x_signal = x[..., :self.signal_dim] 
        x_mask = x[..., self.signal_dim:self.signal_dim + self.mask_dim]

        h_signal = self.signal_mlp(x_signal)
        h_mask = self.mask_mlp(x_mask)

        h = torch.cat([h_signal, h_mask], dim=-1)
        y = self.fusion_mlp(h)
        return y
