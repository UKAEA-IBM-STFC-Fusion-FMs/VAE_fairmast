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
    def __init__(self, encoder_layer_specs, encoded_signal_shape, decoder_layer_specs, vae_specs):
        super().__init__()

        try:
            # Extract specs
            first_layer = encoder_layer_specs["layers"][0]
            if first_layer["type"] != "conv1d":
                raise ValueError("First encoder layer must be conv1d")
            
            self.in_channels = first_layer["params"]["in_channels"]
            self.latent_dim = vae_specs["latent_dim"]
            self.input_length = vae_specs["input_length"]

        except (KeyError, IndexError) as e:
            raise ValueError(f"Missing required specification: {e}")
        
        # =============== Encoder =====================
        
        self.encoder = SequentialBuilder(encoder_layer_specs)
        
        # Find shape after encoding
        self.conv_out_channels = encoded_signal_shape[0]
        self.conv_out_length  = encoded_signal_shape[1]
        conv_out_dim = encoded_signal_shape[0]*encoded_signal_shape[1]
        
        # =============== VAE =====================
        print(f"conv_out_dim {conv_out_dim}")
        self.fc_mu = nn.Linear(conv_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, self.latent_dim)

        # =============== Decoder =====================
        self.fc_decode = nn.Linear(self.latent_dim, conv_out_dim)
        self.decoder = SequentialBuilder(decoder_layer_specs)
        
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        # Add a fully connected layer to the conv1d encoder
        FCE = SequentialBuilder(FullyConnectedEncode(encoded.shape[1],encoded.shape[1]))
        encoded = FCE(encoded)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.fc_decode(z)
        decoded = decoded.view(decoded.size(0), self.conv_out_channels, self.conv_out_length)
        x_recon = self.decoder(decoded)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def loss_function(beta, reconstruction, target, mu, logvar):
    """β-VAE loss function"""
    try:
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')  
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
        total_loss = reconstruction_loss + beta * kl_loss
    except:
        reconstruction_loss = F.mse_loss(reconstruction, target[:, :, :reconstruction.shape[2]], reduction='mean')  
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
        total_loss = reconstruction_loss + beta * kl_loss

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def test_conv1d_vae():
    # Model hyperparameters
    beta = 1
    in_channels = 3
    input_length = 100
    out_channels = 5
    latent_dim = 3
    kernel_size = 7
    stride = 5
    padding = 0

    vae_specs = {
        "beta": beta, 
        "latent_dim": latent_dim, 
        "input_length": input_length
    }
    
    # Encoder layer specs
    encoder_layer_specs = {
        "layers": [
            {
                "type": "conv1d",
                "params": {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": padding
                }
            },
            {
                "type": "relu"
            }
        ]
    }

    decoder_layer_specs = build_decoder_specs_from_encoder_specs(
        encoder_layer_specs,
        in_channels, 
        input_length 
    )

    # Create model
    model = Conv1dVAE(encoder_layer_specs, decoder_layer_specs, vae_specs)

    # Dummy input
    x = torch.randn(4, in_channels, input_length)

    # Forward pass
    x_recon, mu, logvar = model(x)

    # Compute loss
    loss, recon_loss, kl_loss = loss_function(beta, x_recon, x, mu, logvar)

    # Print results
    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_recon.shape)
    print("Latent dim:", mu.shape[1])
    print("Loss:", loss.item())
    print("Reconstruction Loss:", recon_loss.item())
    print("KL Divergence Loss:", kl_loss.item())


if __name__ == "__main__":
    test_conv1d_vae()
