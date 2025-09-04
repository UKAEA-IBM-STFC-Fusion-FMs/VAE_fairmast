import os
import sys
import pickle
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms
)
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import (
    get_mean_shot,
    get_std_shot,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_with_zeros_imputer_transform import (
    FillWithZerosImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.forward_fill_imputer_transform import (
    ForwardFillImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)

from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.models.conv1d_vae_model import Conv1dVAE
from scripts.pipelines.models.conv1d_vae_model import loss_function
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.collate_functions.collate_functions import conv1d_vae_collate_fn

print(f"\nNumber of Cores: {cpu_count()}\n")

# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]

    return train_sh, test_sh, val_sh

def fit_mean_and_std_for_signal_transform( 
                                          train_shots,
                                          output_dir, 
                                          verbose=False,
                                          use_existing=False, 
                                          local=True
                                          ):
    """
    Fit or load mean and std for signal transformation.

    Args:
        output_sub_dir: Directory to save/load fitted parameters
        verbose: Print verbose output
        use_existing: If True, try to load existing fitted parameters instead of re-fitting
    """
    os.makedirs(output_dir, exist_ok=True)

    mean_path = os.path.join(output_dir, "dict_mean_shot.pkl")
    std_path = os.path.join(output_dir, "dict_std_shot.pkl")

    # Try to load existing files if requested
    if use_existing and os.path.exists(mean_path) and os.path.exists(std_path):
        if verbose:
            print("\n\n----------LOADING EXISTING FITTED PARAMETERS----------\n")
            print(f"Loading fitted parameters from: {output_dir}")

        try:
            with open(mean_path, "rb") as f:
                dict_mean_ = pickle.load(f)
            with open(std_path, "rb") as f:
                dict_std_ = pickle.load(f)

            if verbose:
                print(f"Successfully loaded mean and std dictionaries")
                print(f"Mean dict keys: {list(dict_mean_.keys())}")
                print(f"Std dict keys: {list(dict_std_.keys())}")

            return dict_mean_, dict_std_

        except Exception as e:
            if verbose:
                print(f"Error loading existing fitted parameters: {e}")
                print("Falling back to re-fitting")

    if verbose:
        print("\n\n----------TRANSFORM FITTING----------\n")

    preprocessing_train_dataset = MastDataset(
        local=local,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None,
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    dict_std_ = get_std_shot(preprocessing_train_dataset)

    # Save fitted parameters
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir}")

    with open(mean_path, "wb") as f:
        pickle.dump(dict_mean_, f)
    with open(std_path, "wb") as f:
        pickle.dump(dict_std_, f)

    return dict_mean_, dict_std_

def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False
    ):
    
    datasets_ = {"train": None, "val": None, "test": None}
    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if shots[data_set_type]:
            datasets_[data_set_type] = MastDataset(
                local=local_flag,
                shots_list=shots[data_set_type],
                source_signal_list=sources_and_signals,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transforms,
            )
            
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False
    ):
    
    dataloaders_ = {"train": None, "val": None, "test": None}

    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if datasets[data_set_type]:
            dataloaders_[data_set_type] = DataLoader(
                dataset=datasets[data_set_type],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_function,
            )

    return dataloaders_

def create_conv1d_vae_models(
    train_dataloader, 
    beta, 
    channels_factor=5,
    kernel_factor=10,
    padding = 0,
    verbose = False
    ):
    """Create conv1d-VAE models for each signal type"""

    # Get sample batch to determine signal shapes
    sample_batch = next(iter(train_dataloader))
    
    models = {}
    for signal_name, signal_data in sample_batch.items():

        input_length = signal_data[0].shape[-1]  # Last dimension is time
        input_channels = signal_data[0].shape[-2] # Nr. of channels

        if verbose:
            print(
                f"Signal: {signal_name}, Shape: {signal_data.shape}, Input length: {input_length}"
            )

        out_channels = max(1, int(input_channels/channels_factor))
        latent_dim = max(1,out_channels)
        kernel_size = max(1,int(input_length/kernel_factor))
        stride = int(kernel_size/2)
        
        
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
                        "in_channels": input_channels,
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
                        "out_channels": input_channels,
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
    
        model = Conv1dVAE(encoder_layer_specs, decoder_layer_specs, vae_specs)

        models[signal_name] = model

        if verbose:
            print(f"Created BetaVAE for {signal_name}")

    return models

def train_beta_vae_models(
    beta_vae,
    models, 
    device,
    train_dataloader, 
    val_dataloader, 
    output_dir, 
    verbose=False):
    
    """Train β-VAE models for each signal"""


    os.makedirs(output_dir, exist_ok=True)

    # Create optimizers for each model
    optimizers = {}
    for signal_name, model in models.items():
        optimizers[signal_name] = torch.optim.Adam(model.parameters(), lr=SETTINGS.BETA_VAE.lr)

    # Training tracking
    best_losses = {signal_name: float("inf") for signal_name in models.keys()}
    best_model_states = {}
    
    # Loss tracking
    loss_curves = {}
    for signal_name in models.keys():
        loss_curves[signal_name] = {
            'train_total': [],
            'train_recon': [],
            'train_kl': [],
            'val_total': [],
            'val_recon': [],
            'val_kl': []
        }

    for epoch in range(SETTINGS.TRAINING.num_epochs):
        verbose and print(f"\nEpoch {epoch+1}\n")

        # Training phase
        verbose and print("Training phase")
        for signal_name, model in models.items():
            model.train()
            model.to(device)

        train_losses = defaultdict(float)
        train_recon_losses = defaultdict(float)
        train_kl_losses = defaultdict(float)
        train_counts =  defaultdict(float)

        for batch_idx, batch in enumerate(train_dataloader):
            verbose and print(f"Batch idx: {batch_idx}")
                
            for signal_name, data in batch.items():
                model = models[signal_name]
                optimizer = optimizers[signal_name]
                
                for x in data:
                    x = x.to(device)
                    
                    x_recon, mu, logvar = model(x)

                    # Compute loss
                    total_loss, recon_loss, kl_loss = loss_function(beta_vae, x_recon, x, mu, logvar)


                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    train_losses[signal_name] += total_loss.item()
                    train_recon_losses[signal_name] += recon_loss.item()
                    train_kl_losses[signal_name] += kl_loss.item()
                    train_counts[signal_name] += 1
    

        # Validation phase
        val_losses = defaultdict(float)
        val_recon_losses = defaultdict(float)
        val_kl_losses = defaultdict(float)
        val_counts = defaultdict(int)

        for signal_name, model in models.items():
            model.eval()

        verbose and print("\nValidation phase")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                verbose and print(f"Batch idx: {batch_idx}")
                
                for signal_name, data in batch.items():
           
                    model = models[signal_name]

                    for x in data:
                        x = x.to(device)
                        
                        x_recon, mu, logvar = model(x)

                        # Compute loss
                        total_loss, recon_loss, kl_loss = loss_function(beta_vae, x_recon, x, mu, logvar)

                        val_losses[signal_name] += total_loss.item()
                        val_recon_losses[signal_name] += recon_loss.item()
                        val_kl_losses[signal_name] += kl_loss.item()
                        val_counts[signal_name] += 1

        # Store loss curves and print epoch results
        for signal_name in models.keys():
            if  train_counts[signal_name] > 0:
                avg_train_loss = train_losses[signal_name] / train_counts[signal_name]
                avg_train_recon = train_recon_losses[signal_name] / train_counts[signal_name]
                avg_train_kl = train_kl_losses[signal_name] / train_counts[signal_name]
            else: 
                avg_train_loss = float("inf")
                avg_train_recon = float("inf")
                avg_train_kl = float("inf")
            
            if val_counts[signal_name] > 0:
                avg_val_loss = avg_val_loss  = val_losses[signal_name] / val_counts[signal_name]
                avg_val_recon = val_recon_losses[signal_name] / val_counts[signal_name]
                avg_val_kl = val_kl_losses[signal_name] / val_counts[signal_name]
            else:
                avg_val_loss = float("inf")
                avg_val_recon = float("inf")
                avg_val_kl = float("inf")
            

            # Store loss curves
            loss_curves[signal_name]['train_total'].append(avg_train_loss)
            loss_curves[signal_name]['train_recon'].append(avg_train_recon)
            loss_curves[signal_name]['train_kl'].append(avg_train_kl)
            loss_curves[signal_name]['val_total'].append(avg_val_loss)
            loss_curves[signal_name]['val_recon'].append(avg_val_recon)
            loss_curves[signal_name]['val_kl'].append(avg_val_kl)

            if verbose:
                print(
                    f"Signal {signal_name:30s} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

            # Save best model
            if avg_val_loss < best_losses[signal_name]:
                best_losses[signal_name] = avg_val_loss
                best_model_states[signal_name] = models[signal_name].state_dict()

                # Save best model state
                model_path = os.path.join(
                    output_dir, f"best_beta_vae_{signal_name.replace('/', '_')}.pt"
                )
                torch.save(best_model_states[signal_name], model_path)

    return best_model_states, loss_curves


if __name__ == "__main__":

    # Initialize SETTINGS object
    SETTINGS = get_settings("scripts/pipelines/configs/config.json")
    
    mp.set_start_method("spawn", force=True)

    # For common pipeline
    OUTPUT_SUB_FOLDER = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_output/"

    source_signal_list = SETTINGS.DATA.data_names

    # Parameters for window segmentation (no x/y split for VAE)
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,  # 100ms windows
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "min_samples_per_window": SETTINGS.TIME_SEGMENTATION.min_samples_per_window,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        SETTINGS.TRAINING.num_train_samples
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        train_shots,
        OUTPUT_SUB_FOLDER,
        verbose=False,
        use_existing=SETTINGS.BETA_VAE.existing_fitted_params,
        local = SETTINGS.DATA.local
    )

    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [
                ForwardFillImputerTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillWithZerosImputerTransform(),
                SamplingToReferenceTimeTransform(SETTINGS.BETA_VAE.ref_freq),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transforms = ComposeTransforms(
        [
            TruncationTransform(),
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            Conv1dVAETransform(),
        ]
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    test_dataloader = dataloaders_train_val_test["test"]

    # Create β-VAE models
    conv1d_vae_models = create_conv1d_vae_models(
        train_dataloader, 
        SETTINGS.BETA_VAE.beta, 
        verbose = False
    )

    best_model_states, training_loss_curves = train_beta_vae_models(
        SETTINGS.BETA_VAE.beta,
        conv1d_vae_models,
        device,
        train_dataloader,
        val_dataloader,
        OUTPUT_SUB_FOLDER,
        verbose=True
    )

    print("\n\n----------TRAINING-VALIDATION COMPLETE----------")
    print(f"Trained β-VAE models for {len(best_model_states)} signals")
    print(f"Models saved in: {OUTPUT_SUB_FOLDER}")
