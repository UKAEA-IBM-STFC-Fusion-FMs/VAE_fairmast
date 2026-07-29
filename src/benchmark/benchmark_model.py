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
    """The model takes as input a concatenation of latent representations 
       obtained6from multiple pretrained VAEs together with a corresponding validity mask.

        Feature-wise scaling (gamma) and shifting (beta) coefficients. 
        These coefficients modulate the signal features through a 
          
            h = (1 + gamma) * h_signal + beta
        
        This h is then passed to a prediction head that estimates the future latent representation of the target signal.
            

    """
    def __init__(self, SETTINGS):
        super().__init__()


        self.signal_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.signal_layers})

        self.mask_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.mask_layers})
        
        self.fusion_mlp = SequentialBuilder({"layers": SETTINGS.MODEL.end_layers})
        
        # Find signal input-output dimensions from SETTINGS
        self.signal_in_features = None
        self.signal_out_features = None
        for l in SETTINGS.MODEL.signal_layers:
            if l["type"] == "linear":
                if self.signal_in_features ==None: 
                    self.signal_in_features = l["params"]["in_features"]
                self.signal_out_features = l["params"]["out_features"]
        
        # Find mask input-output dimensions from SETTINGS
        self.mask_in_features = None
        self.mask_out_features = None
        for l in SETTINGS.MODEL.mask_layers:
            if l["type"] == "linear":
                if self.mask_in_features ==None: 
                    self.mask_in_features = l["params"]["in_features"]
                self.mask_out_features = l["params"]["out_features"]
        
        # Find end_model input-output dimensions from SETTINGS
        self.end_model_in_features = None
        self.end_model_out_features = None
        for l in SETTINGS.MODEL.end_layers:
            if l["type"] == "linear":
                if self.end_model_in_features == None: 
                    self.end_model_in_features = l["params"]["in_features"]
                self.end_model_out_features = l["params"]["out_features"]
    
        # Check neteork architectures
        if  self.mask_in_features is None or self.mask_out_features is None:
            raise ValueError("Mask input and output dimensions could not be determined. Make sure the mask_model linear")
        
        if  self.signal_in_features is None or self.signal_out_features is None:
            raise ValueError("Signal input and output dimensions could not be determined. Make sure the signal_model linear")
        
        if  self.end_model_in_features is None or self.end_model_out_features is None:
            raise ValueError("The end_model input and output dimensions could not be determined. Make sure the end_model_model linear")

        if  (int(self.mask_out_features/2) != self.signal_out_features) and (int(self.mask_out_features) != self.signal_out_features):
            raise ValueError(f"The mask network should output gamma and beta each one having same size as the input layer of the signal network : {self.signal_out_features}.\
                Current size of mask output is {self.mask_out_features} thus beta and gamma have size {int(self.mask_out_features/2)}")
        
    def forward(self, signal, mask):
       
        h_signal = self.signal_mlp(signal)
        h_mask = self.mask_mlp(mask)
   
        if h_mask.shape[1] == 2*h_signal.shape[1]:
            try:
                gamma, bias = torch.chunk(h_mask, 2, dim=-1)
                h_signal = (1 + gamma) * h_signal + bias
            except Exception as e:
                raise ValueError(f"{e}")
        elif h_mask.shape[1] == h_signal.shape[1]:
             h_signal = h_mask * h_signal 
        else:
            raise ValueError(f"The mask network should output 1) gamma and beta each one having same size as the input layer of the signal network \
                or 2) gamma only. Current size of mask output is {self.mask_out_features} signal size {self.signal_out_features}")
        
        y = self.fusion_mlp(h_signal)
        return y
