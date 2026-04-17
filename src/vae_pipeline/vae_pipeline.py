'''
PyTorch pipeline for training Variational Auto Encoder (VAE) architectures 
for learning the latent space representation of signals from the Mega Ampere Spherical Tokamak (MAST) experiments.

Dataset at: https://huggingface.co/datasets/UKAEA-IBM-STFC/tokamark-dataset

Encoder

The encoder compresses input data x into the latent space z, where dim(z)< dim(x).
The encoder architectures used in the training are:

A stack of conv1d layers.
A stack of conv2d layers.
A series of dense layers.

Decoder

The decoder decompresses z to return x.
The decoder applies the inverse encoder transform in reverse order.
How to use it

Use:
python src/vae_pipeline/vae_pipeline.py --config_file_path src/vae_pipeline/configs/config*.json
'''


import argparse
from collections import defaultdict
import copy
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
import yaml

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fairmast_data_processing.src.MAST_tools.MAST_dataset import MastDataset, CachedDataset

from src.vae_pipeline.utils.utils import (
    read_data_split_csv, ComposeTransforms
)


from fairmast_data_processing.src.MAST_benchmark.tools.transforms.stdscale_transform import StdScalingTransform
from fairmast_data_processing.src.MAST_benchmark.tools.transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from src.vae_pipeline.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)

from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.models.vae_model import loss_function_batch_mean as loss_function
from src.vae_pipeline.transforms.shot_level_transforms.vae_transform import VAETransform
from src.vae_pipeline.transforms.shot_level_transforms.concatenate_signals_transform import ConcatenateSignalsAfterTimeSegmentation
from src.vae_pipeline.collate_functions.collate_functions import WindowsCollate
from src.vae_pipeline.utils.utils import get_train_test_val_shots


def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False,
        cache_data=True,
        other_mast_settings={},
        return_incomplete_shots = False
    ):
    
    datasets_ = {"train": None, "val": None, "test": []}
    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if shots[data_set_type]:
            datasets_[data_set_type] = MastDataset(
                local=local_flag,
                shots_list=shots[data_set_type],
                source_signal_list=sources_and_signals,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transforms,
                other_mast_settings=other_mast_settings,
                return_incomplete_shots = return_incomplete_shots,
                remove_outliers = True
            )
            
    if cache_data:
        datasets_["train"] = CachedDataset(datasets_["train"])
        datasets_["val"]   = CachedDataset(datasets_["val"]) 
        datasets_["test"] = CachedDataset(datasets_["test"]) 
           
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers = False
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
                persistent_workers = persistent_workers
            )

    return dataloaders_


def train_vae_model(
    SETTINGS,
    model,
    optimizer,
    scheduler, 
    device,
    train_dataloader, 
    val_dataloader, 
    output_dir,
    use_amp=True, 
    grad_clip=1,   
    verbose=False,
    ):
    
    # Make directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Signal name
    _, signal_name = SETTINGS.DATA.data_names[0]
    
    # Beta for VAE
    beta = SETTINGS.BETA_VAE.beta
    
    # Training tracking
    best_val_loss = float("inf")
    
    # State tracking
    loss_curves = {'train_total': [],'train_recon': [],'train_kl': [],'val_total': [], 'val_recon': [], 'val_kl': []}
    
    # Clamp tuple
    clamp_logvar = (-50,50)
    epochs_no_improvement = 0
    lr_history = []
    beta_history = []
    
    sub_batch_size = SETTINGS.TRAINING.train_batch_size
    
    model.to(device)
    
    stop_early = False
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        
        if stop_early:
            break
        
        if verbose:
            print(f"\n Epoch {epoch+1} \n")
            print("Training phase")
       
        # Send model to device for training
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Initialize loss variables
        train_losses = train_recon_losses = train_kl_losses = 0.0
        train_counts =  0
        
        # Timing 
        t_0_dataloader = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):

            x = batch["x"]

            if x.numel() == 0:
                continue  # skip empty batch

            if verbose:
                print(f"Batch idx: {batch_idx}")
                print(f"Elapsed time DataLoader {time.time()-t_0_dataloader}")

            x = x.to(device)
            total_tensors = x.size(0)
            
            # Initialiaze gradient
            optimizer.zero_grad(set_to_none=True)
            
            # Timing 
            t_0_model_train = time.time()
            device_process_time = 0
            
            any_backward = False
            for start in range(0, total_tensors, sub_batch_size):
                end = min(start + sub_batch_size, total_tensors)
                
                x_sub_batch = x[start:end]

                # For a 3D signals (i.e., x_sub_batch dimension == 4) we use a conv2d encoder.
                # we must permute the indeces of our tensor to agree with the PyTorch conv2d convention.
                if x_sub_batch.ndim == 4:
                    x_sub_batch = x_sub_batch.permute(0, 3, 1, 2).contiguous() 

                try:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        x_recon, mu, logvar = model(x_sub_batch)
                        loss, recon_loss, kl_loss = loss_function(
                            beta,
                            x_recon,
                            x_sub_batch,
                            mu,
                            logvar,
                            clamp_logvar=clamp_logvar
                        )
                except ValueError as e:
                    # skip this sub-batch
                    print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
                    continue


                if (not torch.isfinite(loss).all()) or (not torch.isfinite(recon_loss).all()) or (not torch.isfinite(kl_loss).all()):
                    print(
                        f"[batch {batch_idx} {start}:{end}] non-finite loss components "
                        f"(loss finite={torch.isfinite(loss).all()}, recon finite={torch.isfinite(recon_loss).all()}, kl finite={torch.isfinite(kl_loss).all()}); skipping sub-batch."
                    )
                    continue
            
                # Update gradients (gradients are summed at each iteration)
                sub_tensors = x_sub_batch.size(0)
                effective_loss = loss * (sub_tensors / float(total_tensors))

                if use_amp:
                    scaler.scale(effective_loss).backward()
                else:
                    effective_loss.backward()
                any_backward = True
                
                device_process_time += (time.time()-t_0_model_train)
                t_0_model_train = time.time()
        
                # Keep track of losses across epochs
                train_losses += loss.item() * sub_tensors
                train_recon_losses += recon_loss.item() * sub_tensors
                train_kl_losses += kl_loss.item() * sub_tensors
                train_counts += sub_tensors
            
            if not any_backward:
                if verbose:
                    print(f"[batch {batch_idx}] no valid sub-batches; skipping optimizer step")
                continue  

            if use_amp:
                scaler.unscale_(optimizer)

            total_norm = None
            if grad_clip and grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                # Guard against NaN/Inf grad norm
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    if verbose:
                        print(f"[batch {batch_idx}] bad grad norm {total_norm}; skipping step")
                    if use_amp:
                        scaler.update()
                    continue
                
            # Update model
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if verbose:
                print(f"Batch processing time {device_process_time:.2f}")      
            t_0_dataloader = time.time()
          
    
        # Validation phase
        val_losses = val_recon_losses = val_kl_losses = 0.0
        val_counts = 0

        model.eval()

        if verbose:
            print("\nValidation phase")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                x = batch["x"]
                
                if x.numel() == 0:
                    continue  # skip empty batch
            
                if verbose:
                    print(f"Batch idx: {batch_idx}")

                x = x.to(device)
                total_tensors = x.size(0)
                
                for start in range(0, total_tensors, sub_batch_size):
                    end = min(start + sub_batch_size, total_tensors)
                
                    x_sub_batch = x[start:end]
                    sub_tensors = x_sub_batch.size(0)
                    
                    # For a 3D signals (i.e., x_sub_batch dimension == 4) we use a conv2d encoder.
                    # we must permute the indeces of our tensor to agree with the PyTorch conv2d.
                    if x_sub_batch.ndim == 4:
                        x_sub_batch = x_sub_batch.permute(0, 3, 1, 2).contiguous() 
                    
                                                        
                    # Compute loss
                    try:
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            x_recon, mu, logvar = model(x_sub_batch)
                            loss, recon_loss, kl_loss = loss_function(
                                beta,
                                x_recon,
                                x_sub_batch,
                                mu,
                                logvar,
                                clamp_logvar=clamp_logvar
                            )
                    except ValueError as e:
                        # skip this sub-batch
                        print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
                        continue
                    
                    val_losses += loss.item() * sub_tensors 
                    val_recon_losses += recon_loss.item() * sub_tensors 
                    val_kl_losses += kl_loss.item() * sub_tensors 
                    val_counts += sub_tensors 

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
        
        # Adapt beta after a few epochs from the start
        if epoch > SETTINGS.TRAINING.patience:
            w1= 0.7
            w2= 1-w1
            if avg_val_kl > 0:
                beta = w1*beta + w2*(0.1*avg_val_recon/avg_val_kl )
            else:
                beta = beta
        
        beta_history.append(beta)
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
                f"Train Loss: {avg_train_loss:.6f}, Train reco: {avg_train_recon}, Train KL: {avg_train_kl}"
            )
            print(
                f"Val Loss: {avg_val_loss:.6f}, Val reco: {avg_val_recon}, Val KL: {avg_val_kl}"
            )
            
        # Save best model
        if  best_val_loss > avg_val_loss:
            best_val_loss= avg_val_loss
            epochs_no_improvement = 0
            
            # Save best model state
            model_path = os.path.join(output_dir, f"best_vae_{signal_name}.pt")
            
            print(f"BEST LOSS FOUND, epoch {epoch}")
            torch.save({
                'model_state_dict': model.state_dict(),        
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, model_path)    
        else:
            epochs_no_improvement +=1 
        
        # Save last model
        model_path = os.path.join(output_dir, f"last_vae_{signal_name}.pt")
        torch.save({
                'model_state_dict': model.state_dict(),        
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, model_path)    

        # Stop early
        if epochs_no_improvement >= SETTINGS.TRAINING.patience and epoch > SETTINGS.TRAINING.min_nr_epochs:
            stop_early = True
                    
        print(f"Training losses {loss_curves['train_total']}")
        print(f"Validation losses {loss_curves['val_total']}")
        print(f"lr history {lr_history}")
        print(f"beta history {beta_history}")
        
        with open(os.path.join(output_dir, 'loss_curves.json'), 'w') as f:
            data = {
                'Loss': loss_curves,
                'lr_history': lr_history,
                'beta_history':beta_history
            }
            json.dump(data, f, indent=4)
               

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
        default = "src/vae_pipeline/configs/config_b_field_tor_probe_saddle_voltage.json",
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
        csv_path = SETTINGS.LOCAL_PATHS.data_split_csv_path
    )
    
    #Get mean and std for signal transformation
    # with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
    #     dict_mean = pickle.load(f)
    # with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
    #     dict_std = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_stats_metadata.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)


    # Signal-level transform map
    if "lcfs" in config_file_name:
        signal_transform_map = {
            var: ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                    ReshapeLcfsTransform(),
                    ImputerTransform()
                ]
            )
            for var in [f"{source}-{signal}" for source, signal in source_signal_list]
        }
    else:
        signal_transform_map = {
            var: ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                    ImputerTransform()
                ]
            )
            for var in [f"{source}-{signal}" for source, signal in source_signal_list]
        }
        

    # Shot-level transform map
    shot_transforms = ComposeTransforms(
        [
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            VAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window)
        ]
    )

    # Prepare datasets
    datasets_train_val = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": []},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local,
        cache_data=SETTINGS.DATA.cache_data,
        return_incomplete_shots = False
    )
    
    vae_collate_fn = WindowsCollate()
    dataloaders_train_val = initialize_dataloaders(
        datasets = datasets_train_val,
        collate_function = vae_collate_fn,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers = num_workers,
        shuffle = True,
        persistent_workers = False
    )
    
    train_dataloader = dataloaders_train_val["train"]
    val_dataloader = dataloaders_train_val["val"]

    # Create conv1d-VAE model
    vae_model = beta_VAE(SETTINGS)
    
    if vae_model is None:
        print("Model error. It was not possible to create your model")
        return
    print(f"Model: \n {vae_model}")
    
    optimizer = torch.optim.Adam(
                vae_model.parameters(), 
                lr = SETTINGS.TRAINING.lr
                )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             optimizer,
             T_0 = SETTINGS.TRAINING.num_epochs,
             T_mult = 1, 
             eta_min = 1e-4
            )
  
    ########### Use this block to continue training from a specific checkpoint ####
    # model_path = "src/vae_pipeline/data/output/conv1d_vae_config_dalpha_voltage_part1/best_vae_filter_spectrometer_dalpha_voltage.pt"
    # print(f"RESUMING TRAINING from {model_path}")
    # checkpoint = torch.load(model_path, map_location='cuda')
    # vae_model.load_state_dict(checkpoint['model_state_dict'])
    # vae_model.to('cuda')
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #######################################################################
  
    # Save model architecture
    with open(os.path.join(output_directory, "model.json"),'w') as f:
       json.dump(
            str(vae_model),
            f,
            indent=4
            )
    
    # Save config file 
    try:
        with open(config_file_path, 'rb') as src, open(os.path.join(output_directory,config_file_name), 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Error copying config file: {e}")
    
    if vae_model:
        start = time.time()
        train_vae_model(
            SETTINGS,
            vae_model,
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
