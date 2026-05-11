'''
For breaf summary check vae_pipeline.py introduction.

New: in this version of the pipeline a masked loss was introduced.

python src/vae_pipeline/vae_pipeline_masked.py --config_file_path src/vae_pipeline/configs/config_flux_loop_flux_test.json --config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_1-1.yaml
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

from MAST_tools.MAST_dataset import MastDataset, CachedDataset
from tokamark.tools.transforms.reshape_lcfs_transform import  ReshapeLcfsTransform
from tokamark.tasks import get_task_metadata
from tokamark.data import initialize_TokaMark_dataset

from src.utils.utils import (
    read_data_split_csv, ComposeTransforms, 
    initialize_datasets, initialize_dataloaders, 
    get_train_test_val_shots, load_task_config
)
from src.common_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)

from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.utils.utils import training_block
from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform

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
    
    # From SETTINGS
    _, signal_name = SETTINGS.DATA.data_names[0]
    beta = SETTINGS.BETA_VAE.beta

    # For tracking
    best_val_loss = float("inf")
    loss_curves = {'train_total': [],'train_recon': [],'train_kl': [],'val_total': [], 'val_recon': [], 'val_kl': []}
    epochs_no_improvement = 0
    lr_history = []
    beta_history = []
    
    # Before starting main loop
    clamp_logvar = (-50,50)
    stop_early = False
    model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Main loop over epochs
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        
        if stop_early:
            break
        
        if verbose:
            print("Training phase")
            print(f"\n Epoch {epoch+1} \n")

        # Set training
        model.train()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Initialize loss variables
        train_losses = train_recon_losses = train_kl_losses = 0.0
        train_counts =  0
        
        # Timing 
        # t_0_dataloader = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            if verbose and batch_idx%100 == 0:
                print(f"Batch idx: {batch_idx}")
                # print(f"Elapsed time DataLoader {time.time()-t_0_dataloader}")

            # Prepare tensors
            x = batch["x"][0]
            try:
                p = next(model.parameters())
                input = x.to(dtype = p.dtype, device = p.device) # real space data
            except Exception as e:
                raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")
            
            # Timing 
            # t_0_model_train = time.time()
            # device_process_time = 0
            
            # Initialiaze gradient
            optimizer.zero_grad(set_to_none=True)

            loss, recon_loss, kl_loss, _, _, _, _ = training_block(
                    input, 
                    model,
                    use_amp, 
                    beta,
                    clamp_logvar)
            
            if loss is None:
                if verbose:
                    print("loss is None skipping this batch")
                continue
            
            if  ((not torch.isfinite(loss).all()) or
                (not torch.isfinite(recon_loss).all()) or 
                (not torch.isfinite(kl_loss).all())):

                print(
                    f"[batch {batch_idx} non-finite loss components or loss is None {loss is None}"
                    f"(loss finite={torch.isfinite(loss).all()}, recon finite={torch.isfinite(recon_loss).all()}, kl finite={torch.isfinite(kl_loss).all()}); skipping sub-batch."
                )
                continue
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

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
                
            # Initialize, Propagate, Step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # device_process_time += (time.time()-t_0_model_train)
            # t_0_model_train = time.time()
        
            # Keep track of losses across epochs
            train_losses += loss.item()
            train_recon_losses += recon_loss.item()
            train_kl_losses += kl_loss.item()
            train_counts += 1
    
            # if verbose:
            #     print(f"Batch processing time {device_process_time:.2f}")      
            # t_0_dataloader = time.time()
    
    
        # Validation phase
        val_losses = val_recon_losses = val_kl_losses = 0.0
        val_counts = 0

        model.eval()

        if verbose:
            print("\nValidation phase")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if verbose and batch_idx%100==0:
                    print(f"Batch idx: {batch_idx}")

                x = batch["x"][0]
                try:
                    p = next(model.parameters())
                    input = x.to(dtype = p.dtype, device = p.device) # real space data
                except Exception as e:
                    raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")
                
                loss, recon_loss, kl_loss, _, _, _, _ = training_block(
                    input, 
                    model,
                    use_amp, 
                    beta,
                    clamp_logvar)
                
                if loss is None:
                    continue
                
                if  ((not torch.isfinite(loss).all()) or
                (not torch.isfinite(recon_loss).all()) or 
                (not torch.isfinite(kl_loss).all())):
                    print(
                        f"[batch {batch_idx} non-finite loss components or loss is None {loss is None}"
                        f"(loss finite={torch.isfinite(loss).all()}, recon finite={torch.isfinite(recon_loss).all()}, kl finite={torch.isfinite(kl_loss).all()}); skipping sub-batch."
                    )
                    continue
                
                val_losses += loss.item() 
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
        default = "",
        type=str,
        help="Path to configuration file for the pipeline.")

    parser.add_argument(
        "--config_task_file_path",
        default="",
        type=str,
        help="Path to configuration YAML task file."
    )
    
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
    
    # Load task config
    config_task_file_path: str = args.config_task_file_path
    print(f"config_task_file_path = {config_task_file_path}")
    
    try:
        config_task = load_task_config(config_task_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Output data folder
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_" + config_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
    
    # Signal names for training 
    source_signal_list = SETTINGS.DATA.data_names

    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

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
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_signals_stats.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)

    # Signal-level transform map
    if "lcfs" in config_file_name:
        signal_transform_map = {
            var: ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                    ReshapeLcfsTransform()
                ]
            )
            for var in [f"{source}-{signal}" for source, signal in source_signal_list]
        }
    else:
        signal_transform_map = {
            var: ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std'])
                ]
            )
            for var in [f"{source}-{signal}" for source, signal in source_signal_list]
        }
        

    # Prepare datasets
    zarr_local_path = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2"
    store_mast_settings = {"base_local_zarr_path":zarr_local_path} if SETTINGS.DATA.local and zarr_local_path else None
    base_datasets = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": []},
        signal_transform_map=signal_transform_map,
        local_flag=SETTINGS.DATA.local,
        store_mast_settings=store_mast_settings
    )
    
    base_train_dataset = base_datasets['train']
    base_val_dataset = base_datasets['val']

    # Tokamark datasets
    model_specific_transform = ModelSpecificTransform()

    train_model_dataset = initialize_TokaMark_dataset(
        dataset=base_train_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows = False,
        verbose=False
    )
    val_model_dataset = initialize_TokaMark_dataset(
        dataset=base_val_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows = False,
        verbose=False
    )

    # Dataloaders
    train_dataloader = DataLoader(
        dataset = train_model_dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =  SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )
    val_dataloader = DataLoader(
        dataset = val_model_dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )

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
            eta_min = 1e-5
            )
    ########### Use this block to continue training from a specific checkpoint ####
    # model_path = "src/vae_pipeline/data/New_VAEs/conv1d_vae_config_flux_loop_flux_p2/best_vae_flux_loop_flux.pt"
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
