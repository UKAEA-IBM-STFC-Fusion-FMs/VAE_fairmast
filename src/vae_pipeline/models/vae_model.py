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
    
from src.vae_pipeline.models.encoder_decoder import EncoderDecoder

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

    def reparameterize(self, mu, logvar, sampling: bool = True):
        if not sampling:
            # Deterministic: use the posterior mean
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sampling: bool = True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sampling=sampling)
        x_recon = self.decode(z)
        return x_recon, mu, logvar



def loss_function_batch_mean(beta, reconstruction, target, mu, logvar, clamp_logvar=(-20, 20)):
    """β-VAE loss function"""
      
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')  
    
    # Guardrails
    logvar = torch.nan_to_num(logvar,nan=0.0,posinf=clamp_logvar[1],neginf=clamp_logvar[0]) 
    
    if clamp_logvar is not None:
       logvar = logvar.clamp(min=clamp_logvar[0], max=clamp_logvar[1])
    else:
        logvar = logvar

    kl_per_dim = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
    kl_loss = kl_per_dim.sum(dim=1).mean() 
     
    total_loss = reconstruction_loss + beta * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss

def loss_function_global_mean(beta, reconstruction, target, mu, logvar, clamp_logvar=(-20.0, 20.0)):
    """β-VAE loss function"""
    
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')  
    
    # Guardrails
    logvar = torch.nan_to_num(logvar,nan=0.0,posinf=clamp_logvar[1],neginf=clamp_logvar[0]) 
        
    if clamp_logvar is not None:
       logvar = logvar.clamp(min=clamp_logvar[0], max=clamp_logvar[1])
    else:
        logvar = logvar

    kl_loss = 0.5 * torch.mean(-1 - logvar + mu.pow(2) + logvar.exp())        
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
       
    An example of config_file_path is: "src/vae_pipeline/data/conv1d_vae_config_coil_current/config_coil_current.json"

    Raises
    ------
    FileNotFoundError
        OR
    KeyError
        This error appears if the encoder in use is not a conv1d. This test routine is meant 
        to be used with conv1d encoders.
        
    """
    # Retrieve SETTINGS for the model configuration
    import argparse
    from src.vae_pipeline.configs.config_setup import get_settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "src/vae_pipeline/data/conv1d_vae_config_coil_current/config_coil_current.json",
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
        nr_channels = SETTINGS.WINDOWsSHAPE.window_channels
        length = SETTINGS.WINDOWsSHAPE.window_length 
    except:
        raise KeyError("Either 'window_channels' or 'window_length' could not be found in SETTINGS")

    # Run tests
    #1
    test_model_reco(model, nr_channels,length)
    #2
    test_model_state_dic_retrieval(model, model_home_directory, nr_channels,length)

    
    
    
        
       