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
from src.pipelines.utils.layer_factory import SequentialBuilder
from src.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from src.pipelines.models.conv1d_encoder_decoder_specs import FullyConnectedLinearRelu

class Conv1dVAE(nn.Module):
    def __init__(self, 
                 conv1d_encoder_layer_specs, 
                 encoded_signal_shape, 
                 conv1d_decoder_layer_specs, 
                 vae_specs
                 ):
        
        super().__init__()

        try:
            self.latent_dim = vae_specs["latent_dim"]
            self.input_length = vae_specs["input_length"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Missing required specification: {e}")
        
        # =============== Encoder =====================
        self.conv1d_encoder = SequentialBuilder(conv1d_encoder_layer_specs)
        
        # Find shape after encoding
        self.conv_out_channels = encoded_signal_shape[0]
        self.conv_out_length  = encoded_signal_shape[1]
        self.conv_out_dim = encoded_signal_shape[0]*encoded_signal_shape[1]
        
        # Add Fully Connected Layer to encoder
        self.FCLlayer_size = int((self.conv_out_dim+self.latent_dim)/2)
        self.FCLencoder = SequentialBuilder(
            FullyConnectedLinearRelu( self.conv_out_dim, self.FCLlayer_size )
            )
        
        # =============== VAE =====================
        self.fc_mu = nn.Linear(self.FCLlayer_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.FCLlayer_size, self.latent_dim)

        # =============== Decoder =====================
        self.FCLdecoder = SequentialBuilder(
            FullyConnectedLinearRelu(self.latent_dim, self.FCLlayer_size)
            )
        self.FCLdecoder2 = SequentialBuilder(
            FullyConnectedLinearRelu(self.FCLlayer_size, self.conv_out_dim)
            )

        self.conv1d_decoder = SequentialBuilder(conv1d_decoder_layer_specs)
        
    def encode(self, x):
        encoded = self.conv1d_encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        encoded = self.FCLencoder(encoded) # Fully connected layer
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # From latent space to input space
        decoded = self.FCLdecoder(z) # Fully connected layer 1
        decoded = self.FCLdecoder2(decoded) # Fully connected layer 2
        # Re-shape
        decoded = decoded.view(decoded.size(0), self.conv_out_channels, self.conv_out_length)
        # Conv1d_decoder
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


def create_conv1d_vae_model(
    SETTINGS,
    dataset, 
    conv1d_vae_collate_fn,
    verbose = False
    ):
    """
    Create a 1D convolutional Variational Autoencoder (Conv1dVAE)
    based on the provided dataset sample and configuration settings.

    This function builds a one-sample DataLoader (batch_size=1) to probe the
    dataset's tensor shape (channels and temporal length). It then uses those
    dimensions together with the `SETTINGS` configuration to construct encoder
    and decoder specifications via `build_conv1d_encoder_decoder`, and finally
    instantiates a `Conv1dVAE` model.

    Parameters
    ----------
    SETTINGS : object
        A configuration object providing the required fields 
            - SETTINGS.BETA_VAE.beta : float
                The β coefficient for the β-VAE KL divergence term.
            - SETTINGS.BETA_VAE.latent_dim : int
                Dimensionality of the latent space.
    dataset : torch.utils.data.Dataset
        A PyTorch MAST dataset 
    conv1d_vae_collate_fn : Callable
        A collate function compatible with the given `dataset` that produces a
        batch where 
    verbose : bool, optional

    Returns
    -------
    Conv1dVAE or None
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False, 
        collate_fn=conv1d_vae_collate_fn)

    # Get one sample from the batch to determine signal shape 
    sample_batch = next(iter(dataloader))
    
    for group_idx, signal_data in sample_batch.items():

        input_length = signal_data.shape[-1]  # Last dimension is time
        input_channels = signal_data.shape[-2] # Nr. of channels
                
        vae_specs = {
            "beta": SETTINGS.BETA_VAE.beta, 
            "latent_dim": SETTINGS.BETA_VAE.latent_dim, 
            "input_length": input_length
        }

        # Encoder layer specs
        try:
            conv1d_encoder_layer_specs, encoded_signal_shape, conv1d_decoder_layer_specs = build_conv1d_encoder_decoder(
                SETTINGS, 
                input_channels, 
                input_length
            )
        except RuntimeError as e:
            print(f"Building encoder error: {e}")
            return None
        
        model = Conv1dVAE(
            conv1d_encoder_layer_specs, 
            encoded_signal_shape,
            conv1d_decoder_layer_specs, 
            vae_specs
            )
        
        break
    
    return model