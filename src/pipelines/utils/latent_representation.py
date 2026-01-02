"""
This script introduces the class LATENTSPACE() to retrieve the
latent space representation of data x. 

The data x is a stack of PyTorch tensors, or it can be a single PyTorch tensor
with the appropriate shape.

1- Class LATENTSPACE(): Initialized with parameter model and called by passing dataset x.
2- Function create_conv1d_vae_model() creates an instance of the model to pass to LATENTSPACE(). 
The architecture of this model must match the architecture of the pre-trained one loaded 
by using load_conv1d_vae_model()
3 Function load_conv1d_vae_model() loads the state dictionary of the pre-trained model into
the model instance created by create_conv1d_vae_model().

"""

import os
import pickle
import sys
from sys import exit
import torch
from torch.utils.data import Dataset
from typing import Callable, Tuple


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

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import StdScalingTransform
from scripts.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import WindowSegmenterTransform
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.collate_functions.collate_functions import Conv1dVAECollate as Conv1dVAECollate
from scripts.pipelines.models.conv1d_vae_model import Conv1dVAE, loss_function
from scripts.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from scripts.pipelines.utils.utils import get_train_test_val_shots
from scripts.pipelines.utils.utils import ComposeTransforms


class LATENTSPACE():
    
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x, select_output: str):
        """Apply model to a stack of PyTorch tensors x and return the latent space representation 
        of such data.

        Parameters
        ----------
        x : PyTorch tensors stack
        select_output: 
            if "z" returns mu (latent space representation of x)
            if "x_rec" returns x_recon (the reconstructed x obtained from its latent space representation)

        """
        try:
            x_recon, mu, logvar = self.model(x)
            
            if select_output == "z":
                return mu
            if select_output == "x_rec":
                return x_recon
        except Exception as e:
            raise RuntimeError(f"Error in loading configuration: {e}")


def create_conv1d_vae_model(
        SETTINGS,
        dataset: Dataset, 
        conv1d_vae_collate_fn: Callable,
        verbose = False
        ):  
    """
    Create a 1D convolutional Variational Autoencoder (Conv1dVAE)
    based on the provided dataset and configuration settings.

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
    
def load_conv1d_vae_model(device, dataset:Dataset, model_path:str, SETTINGS):
    """
    Load the parameters of a Conv1dVAE model.

    This function:
    1. Creates a Conv1dVAE model using `create_conv1d_vae_model`.
    2. Loads the model `state_dict` from the specified directory.


    Parameters
    ----------
    dataset : Dataset
        MAST dataset used to initialize the model architecture.
    model_path : str
        Path to the model (.pt file).
    SETTINGS : object
        Configuration structure loaded from a configurable file.

    Returns
    -------
    Conv1dVAE
        The Conv1dVAE model with loaded parameters.
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model parameters path {model_path} not found.") 
    
    # Create conv1d_vae model
    conv1d_vae_collate_fn = Conv1dVAECollate(SETTINGS.TRAINING.train_batch_size)
    model = create_conv1d_vae_model(
            SETTINGS,
            dataset,
            conv1d_vae_collate_fn
        )
    if model is None:
        print("Model initialization failed")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Model sent to device {device}")
        
    except Exception as e:
        raise RuntimeError(f"Error in loading configuration: {e}")
    
    return model
            
               



############## TEST PART ##############

def create_dataset(SETTINGS):

    source_signal_list = SETTINGS.DATA.data_names

    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,  # 100ms windows
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "verbose": False,
    }

    train_shots, _, val_shots = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = None
    )
    
    # Signal-level transform map
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
        dict_mean = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
        dict_std = pickle.load(f)
        
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_mean[var], dict_std[var]),
                ImputerTransform(),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }
    
    # Shot-level transform map
    shot_transforms = ComposeTransforms(
            [
                WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
                Conv1dVAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window),
            ]
        )

    # Prepare datasets
    dataset = MastDataset(
        source_signal_list = source_signal_list,
        shots_list = val_shots,
        signal_level_transform_map = signal_transform_map,
        shot_level_transform = shot_transforms,
        local = SETTINGS.DATA.local
    )
    return dataset
        
        
def get_data_at(index:int, dataset: Dataset, SETTINGS):
    conv1d_vae_collate_fn = Conv1dVAECollate(SETTINGS.TRAINING.train_batch_size)
    data = dataset[index]
    return conv1d_vae_collate_fn(data)
      
      
if __name__ =="__main__":
    # Determine available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--------------- RUNNING ON GPUs ---------------")
    else:
        device = torch.device("cpu")
        print(f"--------------- RUNNING ON CPUs ---------------")
        
    # Create SETTINGS
    model_path = "scripts/pipelines/data/output/conv1d_vae_config_flux_loop_flux/best_conv1d_vae_flux_loop_flux.pt"
    settings_path = "scripts/pipelines/data/output/conv1d_vae_config_flux_loop_flux/config_flux_loop_flux.json"
    
    SETTINGS = get_settings(settings_path)
     
    # Create MAST dataset
    dataset = create_dataset(SETTINGS)
    
    # Load pre-trained model
    model = load_conv1d_vae_model(device, dataset, model_path, SETTINGS)
    
    # Inference part begins: 
    z_model= LATENTSPACE(model)
    
    data = get_data_at(100, dataset, SETTINGS)
    for group_idx, stacked_tensor in data.items():
        x = stacked_tensor.to(device)
        break # retrieve first x
    
    z = z_model(x,"z")
    print(f"x.shape {x.shape}")
    print(f"z.shape {z.shape}")
    print(f"z: {z}")




