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


class ForecastingModel(nn.Module):
    """
    """
    def __init__(self, SETTINGS):
        super().__init__()
        
        self.signal_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.signal_layers})
        self.mask_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.mask_layers})
        self.end_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.end_layers})
    
    def forward(self, signal, mask):
        h_signal = self.signal_mlp(signal)
        h_mask = self.mask_mlp(mask)
   
        h_signal = h_mask * h_signal 


        return self.end_mlp(h_signal)
