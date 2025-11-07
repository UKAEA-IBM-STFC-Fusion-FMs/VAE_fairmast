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
from scripts.pipelines.utils.layer_factory import SequentialBuilder
from scripts.pipelines.models.conv1d_encoder_decoder_specs import FullyConnectedEncode

class Conv1dVAE(nn.Module):
    def __init__(self, conv1d_encoder_layer_specs, encoded_signal_shape, conv1d_decoder_layer_specs, vae_specs):
        super().__init__()

        try:
            # Extract specs
            first_layer = conv1d_encoder_layer_specs["layers"][0]
            if first_layer["type"] != "conv1d":
                raise ValueError("First encoder layer must be conv1d")
            
            self.in_channels = first_layer["params"]["in_channels"]
            self.latent_dim = vae_specs["latent_dim"]
            self.input_length = vae_specs["input_length"]

        except (KeyError, IndexError) as e:
            raise ValueError(f"Missing required specification: {e}")
        
        # =============== Encoder =====================
        self.conv1d_encoder = SequentialBuilder(conv1d_encoder_layer_specs)
        
        # Find shape after encoding
        self.conv_out_channels = encoded_signal_shape[0]
        self.conv_out_length  = encoded_signal_shape[1]
        conv_out_dim = encoded_signal_shape[0]*encoded_signal_shape[1]
        
        # Add Fully Connected Layer to encoder
        self.FCLencoder = SequentialBuilder(FullyConnectedEncode(conv_out_dim,conv_out_dim))
        
        # =============== VAE =====================
        self.fc_mu = nn.Linear(conv_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, self.latent_dim)

        # =============== Decoder =====================
        self.FCLdecoder = SequentialBuilder(FullyConnectedEncode(self.latent_dim, conv_out_dim))
        self.conv1d_decoder = SequentialBuilder(conv1d_decoder_layer_specs)
        
    def encode(self, x):
        encoded = self.conv1d_encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        encoded = self.FCLencoder(encoded)

        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded =  self.FCLdecoder(z)
        decoded = decoded.view(decoded.size(0), self.conv_out_channels, self.conv_out_length)
        x_recon = self.conv1d_decoder(decoded)
        return x_recon

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

