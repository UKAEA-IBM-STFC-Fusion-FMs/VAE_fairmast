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

from fairmast_tools.MAST_tools.MAST_dataset import MastDataset, CachedDataset

from src.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms
)

from src.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)

from src.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)

from src.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.pipelines.configs.config_setup import get_settings
from src.pipelines.models.vae_model import beta_VAE
from src.pipelines.models.vae_model import loss_function
from src.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from src.pipelines.transforms.shot_level_transforms.concatenate_signals_transform import ConcatenateSignalsAfterTimeSegmentation
from src.pipelines.collate_functions.collate_functions import Conv1dVAECollate
from src.pipelines.utils.utils import get_train_test_val_shots


def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False,
        cache_data=True,
        other_mast_settings={}
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
                other_mast_settings=other_mast_settings
            )
            
    if cache_data:
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


def train_conv1d_vae_model(
    SETTINGS,
    model,
    optimizer,
    scheduler, 
    device,
    train_dataloader, 
    val_dataloader, 
    output_dir, 
    verbose=False,
    ):
    
    # Make directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Signal name
    _, signal_name = SETTINGS.DATA.data_names[0]

    # Training tracking
    best_losses = float("inf")
    best_model_states = {}
    
    # State tracking
    loss_curves = {
        'train_total': [],
        'train_recon': [],
        'train_kl': [],
        'val_total': [],
        'val_recon': [],
        'val_kl': []
    }
    
    epochs_no_improvement = 0
    lr_history = []
        
    stop_early = False
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        if stop_early:
            break
        
        if verbose:
            print(f"\nEpoch {epoch+1}\n")

        # Training phase
        if verbose:
            print("Training phase")
       
        model.to(device)
        model.train()
            
        train_losses = 0
        train_recon_losses = 0
        train_kl_losses = 0
        train_counts =  0
        
        start = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            
            if batch is None:
                continue
            
            if verbose:
                print(f"Batch idx: {batch_idx}")
            if verbose:
                print(f"Elapsed time DataLoader {time.time()-start}")
          
            device_average_process_time = 0
            start_device = time.time()
            for group_idx, stacked_tensor in batch.items():
                x = stacked_tensor.to(device)
                
                x_recon, mu, logvar = model(x)

                # Compute loss
                total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_losses += total_loss.item()
                train_recon_losses += recon_loss.item()
                train_kl_losses += kl_loss.item()
                train_counts += 1
                
                device_average_process_time += (time.time()-start_device)
                start_device = time.time()
               
            if verbose:
                print(f"Batch processing time {device_average_process_time:.2f}")      
            start = time.time()
    
        # Validation phase
        val_losses = 0
        val_recon_losses = 0
        val_kl_losses = 0
        val_counts = 0

        model.eval()

        if verbose:
            print("\nValidation phase")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if verbose:
                    print(f"Batch idx: {batch_idx}")

                for group_idx, stacked_tensor in batch.items():
                    x = stacked_tensor.to(device)
                    
                    x_recon, mu, logvar = model(x)

                    # Compute loss
                    total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)

                    val_losses += total_loss.item()
                    val_recon_losses += recon_loss.item()
                    val_kl_losses += kl_loss.item()
                    val_counts += 1

        # Store loss curves and print epoch results
        if  train_counts > 0:
            avg_train_loss = train_losses / train_counts
            avg_train_recon = train_recon_losses / train_counts
            avg_train_kl = train_kl_losses / train_counts
        else: 
            avg_train_loss = float("inf")
            avg_train_recon = float("inf")
            avg_train_kl = float("inf")
        
        if val_counts > 0:
            avg_val_loss  = val_losses / val_counts
            avg_val_recon = val_recon_losses / val_counts
            avg_val_kl = val_kl_losses / val_counts
        else:
            avg_val_loss = float("inf")
            avg_val_recon = float("inf")
            avg_val_kl = float("inf")
        
        lr_history.append(optimizer.param_groups[0]['lr'])
        scheduler.step()  

        # Store loss curves
        loss_curves['train_total'].append(avg_train_loss)
        loss_curves['train_recon'].append(avg_train_recon)
        loss_curves['train_kl'].append(avg_train_kl)
        loss_curves['val_total'].append(avg_val_loss)
        loss_curves['val_recon'].append(avg_val_recon)
        loss_curves['val_kl'].append(avg_val_kl)

        if verbose:
            print(
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        # Save best model
        if  best_losses - avg_val_loss > SETTINGS.TRAINING.min_increment:
            best_losses = avg_val_loss
            best_model_states = model.state_dict()
            epochs_no_improvement = 0

            # Save best model state
            model_path = os.path.join(
                output_dir, f"best_conv1d_vae_{signal_name}.pt"
            )
            
            torch.save({
                'model_state_dict': model.state_dict(),        
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, model_path)    
        else:
            epochs_no_improvement +=1 
        
        # Stop early
        if epochs_no_improvement >= SETTINGS.TRAINING.patience and epoch > SETTINGS.TRAINING.min_nr_epochs:
            stop_early = True
                    
        print(f"Training losses {loss_curves['train_total']}")
        print(f"Validation losses {loss_curves['val_total']}")
        print(f"lr history {lr_history}")
    
        with open(os.path.join(output_dir, 'loss_curves.json'), 'w') as f:
            data = {
                'Loss': loss_curves,
                'lr_history': lr_history
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
        default = "src/pipelines/configs/config_flux_loop_flux.json",
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
    
    # HPC settings
    num_workers = SETTINGS.TRAINING.num_workers

    # Output data folder
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_" + config_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
    
    # Signal names for training 
    source_signal_list = SETTINGS.DATA.data_names

    # Parameters for window segmentation
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "verbose": False,
    }

    # Create sets of shot IDs for training, testing and validation
    train_shots, _, val_shots = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = None,
        csv_path = "fairmast_tools/metadata/2025-05-12/data_splits.csv"
    )
    
    #Get mean and std for signal transformation
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
        dict_mean = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
        dict_std = pickle.load(f)


    # Signal-level transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_mean[var], dict_std[var]),
                ImputerTransform()
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform map
    if len(SETTINGS.DATA.data_names)>1: # Merge signals
        print("WARNING: current pipeline supports single signal analysis only.\
            All signals in the list will be merged into one, if compatible")
        shot_transforms = ComposeTransforms(
            [
                WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
                ConcatenateSignalsAfterTimeSegmentation(),
                Conv1dVAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window),
            ]
        )
    else:
        shot_transforms = ComposeTransforms(
            [
                WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
                Conv1dVAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window),
            ]
        )

    # Prepare datasets
    datasets_train_val = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": []},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local
    )
    
    conv1d_vae_collate_fn = Conv1dVAECollate(SETTINGS.TRAINING.train_batch_size)
    dataloaders_train_val = initialize_dataloaders(
        datasets=datasets_train_val,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=True
    )
    train_dataloader = dataloaders_train_val["train"]
    val_dataloader = dataloaders_train_val["val"]

    # Create conv1d-VAE model
    conv1d_vae_model = beta_VAE(SETTINGS)
    
    if conv1d_vae_model is None:
        print("Model error. It was not possible to create your model")
        return
    print(f"Model: \n {conv1d_vae_model}")
    
    optimizer = torch.optim.Adam(
                conv1d_vae_model.parameters(), 
                lr = SETTINGS.TRAINING.lr
                )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             optimizer,
             T_0 = SETTINGS.TRAINING.min_nr_epochs,
             T_mult = 1, 
             eta_min = 1e-6
            
            )
  
    ########### Use this block to continue training from a specific checkpoint ####
    # model_path = "src/pipelines/data/output/conv1d_vae_config10_part3/best_conv1d_vae_magnetics-flux_loop_flux.pt"
    # checkpoint = torch.load(model_path)
    # conv1d_vae_model.load_state_dict(checkpoint['model_state_dict'])
    # conv1d_vae_model.to('cuda')
    # optimizer = torch.optim.Adam(
    #         conv1d_vae_model.parameters())
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # optimizer.param_groups[0]['lr'] = SETTINGS.TRAINING.lr
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #######################################################################

    # Save model architecture
    with open(os.path.join(output_directory, "model.json"),'w') as f:
       json.dump(
            str(conv1d_vae_model),
            f,
            indent=4
            )
    
    # Save config file 
    try:
        with open(config_file_path, 'rb') as src, open(os.path.join(output_directory,config_file_name), 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Error copying config file: {e}")
        
    if conv1d_vae_model:
        start = time.time()
        best_model_states, training_loss_curves = train_conv1d_vae_model(
            SETTINGS,
            conv1d_vae_model,
            optimizer,
            scheduler,
            device,
            train_dataloader,
            val_dataloader,
            output_directory,
            verbose=True
        )
        print(f"ELapsed time {time.time() - start}")
        
        print("\n\n----------TRAINING-VALIDATION COMPLETE----------")
        print(f"Models saved in: {output_directory}")
    else:
        print("NO TRAINING: models dictionary is empty.")

if __name__ == "__main__":
    main()
