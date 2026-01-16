import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
        ".."
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from src.pipelines.models.encoder_decoder import EncoderDecoder

class beta_VAE(nn.Module):
    def __init__(self, 
                 SETTINGS
                 ):
        
        super().__init__()
        
        # =============== Encoder-Decoder =====================
        encoder_decoder = EncoderDecoder(SETTINGS)
        self.encoder = encoder_decoder.encoder
        self.decoder = encoder_decoder.decoder
        self.size_before_vae = encoder_decoder.size_before_vae 

        # =============== VAE =====================
        self.latent_dim = SETTINGS.get("BETA_VAE","latent_dim")
        if self.latent_dim is None:
            raise KeyError("latent dimension not found in the SETTINGS")
        
        self.fc_mu = nn.Linear(self.size_before_vae, self.latent_dim)
        self.fc_logvar = nn.Linear(self.size_before_vae, self.latent_dim)

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def loss_function(beta, reconstruction, target, mu, logvar):
    """β-VAE loss function"""
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')  
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss


if __name__ == "__main__":
    import argparse
    from src.pipelines.configs.config_setup import get_settings
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "src/pipelines/configs/config_flux_loop_flux.json",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    config_file_path = args.config_file_path
    config_file_name = os.path.basename(config_file_path)
    
    # Load configuration from JSON file
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
        try:
            SETTINGS = get_settings(config_file_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
         
    model = beta_VAE(SETTINGS)
    print(model)
    
    # Create a synthetic signal with correct shape
    input_length = None
    in_channels = None
    if not getattr(SETTINGS.TIME_SEGMENTATION, "targeted_time_stamps_per_window", None):
        print("Conv1d (time) input_length unresolved")
    else:
        input_length = SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window
    
    if not getattr(SETTINGS.CONV1dENCODER, "conv1d_in_channels", None):
        print("Conv1d conv1d_in_channels unresolved")
    else:
        in_channels = SETTINGS.CONV1dENCODER.conv1d_in_channels
 
    if in_channels is not None and input_length is not None:
        x = torch.randn(in_channels, input_length)
        x = x.unsqueeze(0)
        x_recon, mu, logvar = model(x)
        
        rms = torch.sqrt(torch.mean((x - x_recon) ** 2))
        print(rms)

    
    
    
        
       