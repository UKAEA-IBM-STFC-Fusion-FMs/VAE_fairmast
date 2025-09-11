import math
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
        self.conv_out_channels, self.conv_out_length,_,_ = compute_conv_output_dim(
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


def conv1d_out_len(L_in, k, s=1, p=0, d=1):
    return math.floor((L_in + 2*p - d*(k-1) - 1) / s) + 1

def compute_conv_output_dim(in_channels, input_length, layer_specs):
    """Compute output dimensions after Conv1d layers."""
    current_channels = in_channels
    current_length = input_length
    
    lengths = [input_length]
    channels = [in_channels]
    
    for spec in layer_specs["layers"]:
        if spec["type"] == "conv1d":
            params = spec.get("params", {})
            kernel_size = params.get("kernel_size", 1)
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)
            out_channels = params.get("out_channels", current_channels)

            current_length =  conv1d_out_len(current_length, kernel_size, stride, padding)
            current_channels = out_channels
            
            lengths.append(current_length)
            channels.append(current_channels)

    return current_channels, current_length, lengths, channels

def convt1d_needed_output_padding(L_in, L_out_target, k, s=1, p=0, d=1):
    # L_out_target = desired output of convtranspose
    # L_in = input length to convtranspose (which equals encoder's L_out at that stage)
    base = (L_in - 1) * s - 2*p + d*(k-1) + 1
    op = L_out_target - base
    return op

def build_decoder_layers_from_encoder_specs(encoder_layer_specs, input_channels, input_length):
    """
    encoder_layer_specs: dictionary specifying encoder layers.
    input_channels: original number of channels.
    input_length: original input time length (L0).
    """
    
    # Pass 1: compute encoder output dimensions
    _,_,enc_len,enc_c = compute_conv_output_dim(input_channels, input_length, encoder_layer_specs)
    # enc_len[0] original input length, enc_len[i] output length from i-th conv1d layer
    # enc_c[0] original nr of input channels, enc_c[i] output nr. channels from i-th conv1d layer
    
    # Pass 2: build decoder layers in reverse
    dec_layers = []

    # Reverse loop over encoder layers    
    i=len(enc_len)-1
    for spec in reversed(encoder_layer_specs["layers"]):
        
        if spec["type"] == "conv1d":
            params = spec.get("params", {})
            k = params.get("kernel_size", 1)
            s = params.get("stride", 1)
            p = params.get("padding", 0)
            d = params.get("dilation", 1)
            
           
            in_len = enc_len[i] # current length
            target_len = enc_len[i-1] # we want to get back to this length

            # op: output_padding is additional size added to one side of the output shape
            # It is only used to find output shape, but does not actually add zero-padding to output.
            op = convt1d_needed_output_padding(in_len, target_len, k, s, p, d)
            if not (0 <= op <= s - 1):
                op = max(0, min(s - 1, op))
            layer = nn.ConvTranspose1d(
                in_channels=enc_c[i],
                out_channels=enc_c[i-1],   # mirror channels
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                output_padding=op
            )
            dec_layers.append(layer)
            i -= 1
           
    return dec_layers

def build_decoder_specs_from_encoder_specs(encoder_layer_specs, input_channels, input_length):
    """
    encoder_layer_specs: dictionary specifying encoder layers.
    input_channels: original number of channels.
    input_length: original input time length (L0).
    """
    dec_layers = build_decoder_layers_from_encoder_specs(
        encoder_layer_specs, 
        input_channels, 
        input_length
        )
    
    decoder_layer_specs = {"layers": []}
    
    for layer in dec_layers:
        if isinstance(layer, nn.ConvTranspose1d):
            # Add ConvTranspose1d spec
            spec = {
                "type": "conv_transpose1d",
                "params": {
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size[0],   # tuple → int
                    "stride": layer.stride[0],
                    "padding": layer.padding[0],
                    "dilation": layer.dilation[0],
                    "output_padding": layer.output_padding[0],
                }
            }
            decoder_layer_specs["layers"].append(spec)
            
            # Add ReLU spec after each ConvTranspose1d
            decoder_layer_specs["layers"].append({"type": "relu"})
    
    return decoder_layer_specs

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

    # Decoder layer specs
    # decoder_layer_specs = {
    #     "layers": [
    #         {
    #             "type": "conv_transpose1d",
    #             "params": {
    #                 "in_channels": out_channels,
    #                 "out_channels": in_channels,
    #                 "kernel_size": kernel_size,
    #                 "stride": stride,
    #                 "padding": padding
    #             }
    #         },
    #         {
    #             "type": "relu"
    #         }
    #     ]
    # }
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
