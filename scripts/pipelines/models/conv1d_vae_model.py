import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.pipelines.models.layer_factory import SequentialBuilder


class Conv1dVAE(nn.Module):
    def __init__(self, encoder_layer_specs, decoder_layer_specs, vae_specs):
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
        self.conv_out_channels, self.conv_out_length = _compute_conv_output_dim(
            self.in_channels,
            self.input_length, 
            encoder_layer_specs)
        conv_out_dim = self.conv_out_channels * self.conv_out_length 
        
        # =============== VAE =====================
        self.fc_mu = nn.Linear(conv_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, self.latent_dim)

        # =============== Decoder =====================
        self.fc_decode = nn.Linear(self.latent_dim, conv_out_dim)
        self.decoder = SequentialBuilder(decoder_layer_specs)
        
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
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
    masked_target = target[:, :, :reconstruction.shape[2]]
    reconstruction_loss = F.mse_loss(reconstruction, masked_target, reduction='mean')        
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def _compute_conv_output_dim(in_channels, input_length, layer_specs):
    """Compute output dimensions after Conv1d layers."""
    current_length = input_length
    current_channels = in_channels

    for spec in layer_specs["layers"]:
        if spec["type"] == "conv1d":
            params = spec.get("params", {})
            kernel_size = params.get("kernel_size", 1)
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)
            out_channels = params.get("out_channels", current_channels)

            current_length = (current_length - kernel_size + 2 * padding) // stride + 1
            current_channels = out_channels

    return current_channels, current_length


def test_conv1d_vae():
    # Model hyperparameters
    beta = 1
    in_channels = 10
    input_length = 100
    out_channels = 5
    latent_dim = 3
    kernel_size = 10
    stride = 5
    padding = 0

    vae_specs = {
        "beta": beta, 
        "latent_dim": latent_dim, 
        "input_length": input_length
    }
    
    # Encoder layer specs (new format)
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

    # Decoder layer specs (new format)
    decoder_layer_specs = {
        "layers": [
            {
                "type": "conv_transpose1d",
                "params": {
                    "in_channels": out_channels,
                    "out_channels": in_channels,
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
