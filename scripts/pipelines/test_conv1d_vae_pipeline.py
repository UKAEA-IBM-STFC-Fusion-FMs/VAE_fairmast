import json
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

from scripts.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from scripts.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms, load_models, to_dict
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
from scripts.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform

from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.models.conv1d_vae_model import Conv1dVAE
from scripts.pipelines.models.conv1d_vae_model import loss_function
from scripts.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.collate_functions.collate_functions import Conv1dVAECollate

from scripts.pipelines.beta_vae_pipeline import visualize_beta_vae_results
# Determine device to train on
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--------------- RUNNING ON GPUs ---------------")
else:
    device = torch.device("cpu")
    print(f"--------------- RUNNING ON CPUs ---------------")
    

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
    datasets_["train"] = CachedDataset(datasets_["train"])
    datasets_["val"]   = CachedDataset(datasets_["val"])
    datasets_["test"]  = CachedDataset(datasets_["test"])   
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
    SETTINGS,
    train_dataloader, 
    verbose = False
    ):
    """Create conv1d-VAE models for each signal type"""
    
    # Initalize models
    models = {}
    
    # Get sample batch to determine signal shapes  
    sample_batch = next(iter(train_dataloader))
    
    for signal_name, groups in sample_batch.items():
        for group_idx, signal_data in groups.items():

            input_length = signal_data.shape[-1]  # Last dimension is time
            input_channels = signal_data.shape[-2] # Nr. of channels

            if verbose:
                print(
                    f"Signal: {signal_name}, Shape: {signal_data.shape}, Input length: {input_length}"
                )
                    
            vae_specs = {
                "beta": SETTINGS.BETA_VAE.beta, 
                "latent_dim": SETTINGS.BETA_VAE.latent_dim, 
                "input_length": input_length
            }
    
            # Encoder layer specs
            print(f"signal_name {signal_name}")
            try:
                conv1d_encoder_layer_specs, encoded_signal_shape, conv1d_decoder_layer_specs = build_conv1d_encoder_decoder(
                    SETTINGS, 
                    input_channels, 
                    input_length
                )
            except ValueError as e:
                print(f"Building encoder error: {e}")
                return models

            model = Conv1dVAE(conv1d_encoder_layer_specs, 
                                encoded_signal_shape,
                                conv1d_decoder_layer_specs, 
                                vae_specs)

            models[signal_name] = model

            if verbose:
                print(f"Created conv1dVAE for {signal_name}")
            break

    return models
    
def test_model(source, signal_name, output_dir, SETTINGS):
    model_path = os.path.join(output_dir, "best_conv1d_vae_magnetics-" + signal_name + ".pt")
     
    if not os.path.exists(model_path):
        print(f"{model_path} not found")
        return
    else:
        output_directory = os.path.dirname(model_path)

    # HPC settings for CPUs only
    num_workers = SETTINGS.TRAINING.num_workers

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
        SETTINGS.TRAINING.num_val_samples
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        train_shots,
        output_directory,
        verbose=False,
        use_existing=True,
        local = SETTINGS.DATA.local
    )

    model_dictionary = load_models(SETTINGS.DATA.data_names, 
                                   SETTINGS.LOCAL_PATHS.joblib_directory)
    
    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                ForwardFillImputerTransform(),
                ImputerTransform(model_dictionary["imputer"][var], 
                         SETTINGS.LOCAL_PATHS.average_values_file_path),
                StdScalingTransform(dict_mean[var], dict_std[var]),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transforms = ComposeTransforms(
        [
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
    
    signals_to_collate = [f"{source}-{signal}" for source, signal in source_signal_list]
    conv1d_vae_collate_fn = Conv1dVAECollate(signals_to_collate, SETTINGS.TRAINING.training_batch_size)
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=False # Keep it False since the order need to be deterministic for later analysis
    )
    test_dataloader = dataloaders_train_val_test["test"]
    val_dataloader = dataloaders_train_val_test["val"]
    
    # Create conv1d-VAE models
    model = create_conv1d_vae_models(
        SETTINGS,
        val_dataloader, 
        verbose = False
    )

    model = model[source + "-" + signal_name]
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loss_vs_batch = []
    best_loss = float("inf")
    best_batch_idx  = -1
    best_group_idx = -1
    
    this_signal = "magnetics-flux_loop_flux"
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(val_dataloader):
            print(f"Batch idx {batch_idx}")

            for signal_name, groups in batch.items():
                # Do it for only one signal
                if signal_name !=  this_signal:
                    continue
                
                for group_idx, stacked_tensor in groups.items():
                    
                    x = stacked_tensor.to(device)
                    x_recon, mu, logvar = model(x)

                    # Compute loss
                    total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)
                    loss_vs_batch.append(total_loss.item())  
                    
                    if total_loss.item() < best_loss:
                        best_loss = total_loss.item()
                        best_batch_idx  = batch_idx
                        best_group_idx = group_idx

    
    print(f"Best shot id and group idx= {best_batch_idx}, {best_group_idx}")
    
    
    # Recover best batch
    batch_size = SETTINGS.TRAINING.dataloader_batch_size
    start_idx = best_batch_idx * batch_size
    end_idx = start_idx + batch_size

    # fetch items directly from dataset
    dataset = datasets_train_val_test["val"]
    batch_items = []
    for i in range(start_idx, end_idx):
        try:
            batch_items.append(dataset[i])
        except Exception as e:
            # dataset might not include all items in range(start_idx, end_idx)
            break
 
    recovered_batch = conv1d_vae_collate_fn(batch_items)
    
    x_best_input = None
    x_best_recon = None
    best_loss = float("inf")

    with torch.no_grad():
        for signal_name, groups in recovered_batch.items():
            
            # Do it for only one signal
            if signal_name !=  this_signal:
                continue
            
            for group_idx, stacked_tensor in groups.items():
                if group_idx != best_group_idx:
                    continue
                
                x = stacked_tensor.to(device)
                # Iterate over each tensor in the stack
                for i in range(x.shape[0]):
                    x_i = x[i].unsqueeze(0)  # Add batch dimension
                    x_recon_i, mu_i, logvar_i = model(x_i)

                    total_loss, recon_loss, kl_loss = loss_function(
                        SETTINGS.BETA_VAE.beta, x_recon_i, x_i, mu_i, logvar_i
                    )
                    loss_value = total_loss.item()
                    
                    if loss_value < best_loss:
                        best_loss = loss_value
                        x_best_input = x_i
                        x_best_recon = x_recon_i
    try:
        with open(os.path.join(output_directory , 'test_loss.json'), 'w') as f:
            data = {
                'loss_vs_batch':  loss_vs_batch,
                'best_loss': best_loss,
                'input' : x_best_input.cpu().flatten().numpy().tolist(),
                'reconstructed': x_best_recon.cpu().flatten().numpy().tolist()
            }
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"{e}")
        
    try:
        if x_best_input is not None and x_best_recon is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(x_best_input.cpu().flatten().numpy().tolist(), label="Original", lw=2)
            plt.plot(x_best_recon.cpu().flatten().numpy().tolist(), label="Reconstructed", lw=2, linestyle="--")
            plt.title(f"Best Reconstruction")
            plt.legend()
            plt.show()
            plt.savefig(output_dir+ "best_reconstruction.pdf", dpi=300, bbox_inches='tight')
        else:
            print("No valid reconstruction found.") 
    except Exception as e:
        print(f"{e}")
    
    try:
        x_in = x_best_input.squeeze(0)   # shape: (n_channels, n_length)
        x_re = x_best_recon.squeeze(0)  # shape: (n_channels, n_length)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        im0 = axes[0].imshow(x_in, aspect='auto', cmap='viridis', origin='lower')
        axes[0].set_title("Original")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Channel")

        im1 = axes[1].imshow(x_re, aspect='auto', cmap='viridis', origin='lower')
        axes[1].set_title(f"Reconstructed (loss={best_loss:.4f})")
        axes[1].set_xlabel("Time")

        fig.colorbar(im1, ax=axes.ravel().tolist(), location='right', shrink=0.8, label='Amplitude')
        plt.show()
        plt.savefig(output_dir+"best_reconstruction_image.pdf", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"{e}")


def correlations(data, reco):
    """Compute time correlations for each feature 
    in data-reco pairs

    Parameters
    ----------
    data : tensor
        [batch, features, time]
    reco : _type_
        [batch, features, time]

    Returns
    -------
    Tensor or time correlations
        [batch, features]
    """
    # Subtract mean along time axis
    input_diff = data - data.mean(dim=-1, keepdim=True)
    reco_diff = reco - reco.mean(dim=-1, keepdim=True)

    # Compute numerator and denominator along time axis
    numerator = torch.sum(input_diff * reco_diff, dim=-1)  # [batch, features]
   
    denominator = torch.sqrt(torch.sum(input_diff ** 2, dim=-1) * torch.sum(reco_diff ** 2, dim=-1))  # [batch, features]

    corr = numerator / denominator  # [batch, features]
    
    return corr

def absolute_relative_errors(data, reco, eps = 1e-8):
    """Compute mean absolute relative error for each feature 
    in data-reco pairs

    Parameters
    ----------
    data : tensor
        [batch, features, time]
    reco : _type_
        [batch, features, time]

    Returns
    -------
    Tensor or average absolute errors
        [batch, features]
    """

    # Compute absolute relative error along time axis
    abs_rel_error = torch.abs(data - reco) / (torch.abs(data) + eps)  # [batch, features, time]

    # Mean over time dimension
    features_abs_rel_error = abs_rel_error.mean(dim=-1)  # [batch, features]
    samples_abs_rel_error = features_abs_rel_error.mean(dim =-1)
    min_vals, min_indices = torch.min(samples_abs_rel_error, dim = 0) 

    return mean_abs_rel_error, min_vals, min_indices

if __name__ == "__main__":
    
    SETTINGS = get_settings("scripts/pipelines/configs/config4.json")
    
    output_dir = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_config4/"

    source, signal_name = SETTINGS.DATA.data_names[0]

    test_model(source, signal_name, output_dir, SETTINGS)