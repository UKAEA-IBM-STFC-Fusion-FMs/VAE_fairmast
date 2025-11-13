import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import time

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
from scripts.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from scripts.pipelines.models.conv1d_vae_model import loss_function
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.collate_functions.collate_functions import Conv1dVAECollate

def get_train_test_val_shots(
    max_index_for_train=None,
    max_index_for_val = None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index_for_train:
        train_set = train_sh[0:max_index_for_train]
        val_set = val_sh[0:max_index_for_val]

    return train_set, val_set

def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False
    ):
    
    datasets_ = {"train": None, "val": None}
    data_set_types = ["train", "val"]
    
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
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers=True
    ):
    
    dataloaders_ = {"train": None, "val": None}

    data_set_types = ["train", "val"]
    
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
    dataloader, 
    verbose = False
    ):
    """Create conv1d-VAE models for each signal type"""
    
    # Initalize models
    models = {}
    
    # Get sample batch to determine signal shapes  
    sample_batch = next(iter(dataloader))
    
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

def stop_early(val_losses, min_nr_epochs, patience=5, slope_threshold=1e-4):
    """
    Stop early if validation loss has plateaued or the trend slope is very small.

    Args:
        val_losses: list of floats (validation losses)
        patience: number of recent epochs to check
        slope_threshold: minimum slope magnitude to consider ongoing learning
        min_nr_epochs: minimum number of epochs
    """
    if len(val_losses) < patience or len(val_losses) <= min_nr_epochs:
        return False
    
    try:
        # Check if trend has flattened (slope logic)
        y = np.array(val_losses[-patience:])
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]  # linear regression slope
        flat_trend = abs(slope) < slope_threshold
    except:
        return False
        
    return flat_trend

def train_conv1d_vae_models(
    SETTINGS,
    models, 
    device,
    train_dataloader, 
    val_dataloader, 
    output_dir, 
    verbose=False):
    
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
    
    epochs_no_improvement = 0
    stop_early = False
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        verbose and print(f"\nEpoch {epoch+1}\n")

        # Training phase
        verbose and print("Training phase")
        for signal_name, model in models.items():
            model.to(device)
            model.train()
            
        train_losses = defaultdict(float)
        train_recon_losses = defaultdict(float)
        train_kl_losses = defaultdict(float)
        train_counts =  defaultdict(float)
        
        start = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            verbose and print(f"Batch idx: {batch_idx}")
            verbose and print(f"Elapsed time DataLoader {time.time()-start}")
            
            device_average_process_time = 0
            
            for signal_name, groups in batch.items():
                model = models[signal_name]
                optimizer = optimizers[signal_name]
                
                start_device = time.time()
                for group_idx, stacked_tensor in groups.items():
                    x = stacked_tensor.to(device)
                    
                    x_recon, mu, logvar = model(x)

                    # Compute loss
                    total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)

                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    train_losses[signal_name] += total_loss.item()
                    train_recon_losses[signal_name] += recon_loss.item()
                    train_kl_losses[signal_name] += kl_loss.item()
                    train_counts[signal_name] += 1
                    
                    device_average_process_time += (time.time()-start_device)
                    start_device = time.time()
            verbose and print(f"Device processing time per single data {device_average_process_time/train_counts[signal_name]:.4f}")  
            verbose and print(f"Device processing time all data in batch {device_average_process_time:.2f}")      
            start = time.time()
    
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
                
                for signal_name, groups in batch.items():
           
                    model = models[signal_name]

                    for group_idx, stacked_tensor in groups.items():
                        x = stacked_tensor.to(device)
                        
                        x_recon, mu, logvar = model(x)

                        # Compute loss
                        total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)

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
                avg_val_loss  = val_losses[signal_name] / val_counts[signal_name]
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
                epochs_no_improvement = 0
                best_losses[signal_name] = avg_val_loss
                best_model_states[signal_name] = models[signal_name].state_dict()

                # Save best model state
                model_path = os.path.join(
                    output_dir, f"best_conv1d_vae_{signal_name.replace('/', '_')}.pt"
                )
                torch.save(best_model_states[signal_name], model_path)
            else:
                epochs_no_improvement +=1  
                if epochs_no_improvement == SETTINGS.TRAINING.patience:
                    stop_early = True
        print(f"Training losses {loss_curves[signal_name]['train_total']}")
        print(f"Validation losses {loss_curves[signal_name]['val_total']}")
        
        if stop_early:
            break
    
        with open(os.path.join(output_dir, 'loss_curves.json'), 'w') as f:
            data = {
                'vae': to_dict(SETTINGS.BETA_VAE),
                'conv1d': to_dict(SETTINGS.CONV1D),
                'training': to_dict(SETTINGS.TRAINING),
                'Loss': loss_curves
            }
            json.dump(data, f, indent=4)
                
    return best_model_states, loss_curves


def main():
    mp.set_start_method("spawn", force=True)

    # Determine device to train on
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--------------- RUNNING ON GPUs ---------------")
    else:
        device = torch.device("cpu")
        print(f"--------------- RUNNING ON CPUs ---------------")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "scripts/pipelines/configs/config.json",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    config_file_path = args.config_file_path
    config_file_name = os.path.basename(config_file_path)
    
    # Load configuration from JSON file
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
        try:
            SETTINGS = get_settings(config_file_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
            return 
    
    # HPC settings for CPUs only
    num_workers = SETTINGS.TRAINING.num_workers

    # Output data folder
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_" + config_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print( f"output_directory = {output_directory}")
    
    # Signal names
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
        "tergeted_time_stamp_per_window": SETTINGS.TIME_SEGMENTATION.tergeted_time_stamp_per_window,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, val_shots = get_train_test_val_shots(
        SETTINGS.TRAINING.num_train_samples,
        SETTINGS.TRAINING.num_val_samples
    )
    
    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                ImputerTransform()
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
        shots={"train": train_shots, "val": val_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local
    )
    
    signals_to_collate = [f"{source}-{signal}" for source, signal in source_signal_list]
    conv1d_vae_collate_fn = Conv1dVAECollate(signals_to_collate, SETTINGS.TRAINING.train_batch_size)
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]

    # Create conv1d-VAE models
    conv1d_vae_models = create_conv1d_vae_models(
        SETTINGS,
        val_dataloader, 
        verbose = False
    )
    
    # # Use this block to load a saved model
    # model_path = "scripts/pipelines/data/output/conv1d_vae_config10_new_part1/best_conv1d_vae_magnetics-flux_loop_flux.pt"
    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # conv1d_vae_models["magnetics-flux_loop_flux"].load_state_dict(state_dict)
    # for model in conv1d_vae_models.values():
    #     model.to(device)

    #Save model architectures
    with open(os.path.join(output_directory, "models.json"),'w') as f:
       json.dump(
            {k: str(v) for k, v in conv1d_vae_models.items()},
                f,
                indent=4
            )
    
    # Save config file 
    try:
        with open(config_file_path, 'rb') as src, open(os.path.join(output_directory,config_file_name), 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Error copying config file: {e}")
        
    if conv1d_vae_models:
        start = time.time()
        best_model_states, training_loss_curves = train_conv1d_vae_models(
            SETTINGS,
            conv1d_vae_models,
            device,
            train_dataloader,
            val_dataloader,
            output_directory,
            verbose=True
        )
        print(f"ELapsed time {time.time() - start}")
        
        print("\n\n----------TRAINING-VALIDATION COMPLETE----------")
        print(f"Trained β-VAE models for {len(best_model_states)} signals")
        print(f"Models saved in: {output_directory}")
    else:
        print("NO TRAINING: models dictionary is empty.")

if __name__ == "__main__":
    main()
