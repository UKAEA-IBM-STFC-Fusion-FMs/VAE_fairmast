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


def get_model_state(model:beta_VAE, model_path:str):
    """Retrieve the state dictionary of a model saved at model_path

    Parameters
    ----------
    model : beta_VAE
        This is an empty model initialized with the expected architecture, i.e., beta_VAE.
    model_path : str
        path to the saved model. The saved model MUST contain a model_state_dict key.
    """
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    

def test_model_reco(model, in_channels, input_length):
    """Generate rnd tensor of shape(in_channels, input_length).
       Reconstruct the tensor with the model and print the RMS-error.

    Parameters
    ----------
    model: beta_VAE
        Model for beta-VAE.
    in_channels : int
        Number of channels in the tensor.
    input_length : int
        Length of each channel.

    Returns
    -------
    Tuple of tensors, one for the generated input signal and one for the reconstructed.
    """
   
    x = torch.randn(in_channels, input_length)
    x = x.unsqueeze(0)
    x_recon, mu, logvar = model(x)
    
    rms = torch.sqrt(torch.mean((x - x_recon) ** 2))
    print(rms)

    return x, x_recon
    


def test_model_state_dic_retrieval(model, model_home_directory, nr_channels, length):
    
    for name in os.listdir(model_home_directory):
        model_path = os.path.join(model_home_directory, name)
        if os.path.isfile(model_path) and name.endswith(".pt"):
            break
            
    model = get_model_state(model, model_path)
    x = torch.randn(nr_channels, length)
    x = x.unsqueeze(0)
    mu, logvar = model.encode(x)
    x_recon = model.decode(mu)

    print(f"Input: {x}")
    print(f"Latent representation: {mu}")
    print(f"Decoded: {x_recon}")    
        
        
if __name__ == "__main__":
    """
    Generate and test a beta_VAE model created by using the config SETTINGS for a given MAST signal.
    SETTINGS is created automatically by passing the path to the config.json file when calling 
       
       ```python vae_model.py --config_file_path path_to_config_file.json```
       
    An example of config_file_path is: "src/pipelines/data/conv1d_vae_config_coil_current/config_coil_current.json"

    Raises
    ------
    FileNotFoundError
        OR
    KeyError
        
    """
    # Retrieve SETTINGS for the model configuration
    import argparse
    from src.pipelines.configs.config_setup import get_settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "src/pipelines/data/conv1d_vae_config_coil_current/config_coil_current.json",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    config_file_path = args.config_file_path
    config_file_name = os.path.basename(config_file_path)
    model_home_directory = os.path.dirname(config_file_path)

    # Load configuration from JSON file
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
        try:
            SETTINGS = get_settings(config_file_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
    
    # Create model for the retrieved SETTINGS
    model = beta_VAE(SETTINGS)
    
    # Get signal number of channels and length
    try:
        nr_channels = SETTINGS.ENCODER.layers[0]["params"]["in_channels"]
        length = SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window 
    except:
        raise KeyError("Either 'in_channels' or 'targeted_time_stamps_per_window' could not be found in SETTINGS")

    # Run tests
    #1
    test_model_reco(model, nr_channels,length)
    #2
    test_model_state_dic_retrieval(model, model_home_directory, nr_channels,length)

    
    
    
        
       