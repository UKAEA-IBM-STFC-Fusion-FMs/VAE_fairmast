""" 
PyTorch pipeline to evaluate trained VAEs over tasks defined in tokamark.
For more details on the benchmark study see arXiv:2602.10132 

RUN:
python src/benchmark/benchmark_pipeline.py 
--config_benchmark_file_path path_to_json_benchmark_file 
--config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_1-1.yaml


DATA INGESTION:

This pipeline enforces a strict one‑to‑one correspondence between 
configured signals and their associated Variational Autoencoder (VAE) models at ingestion time. 
The configuration defines three categories of signals: inputs, actuators, and outputs, 
each of which may require compression via a dedicated VAE.

Input signals:
    All input signals must have corresponding input VAEs.
    Input signals without VAEs are not allowed.

Actuator signals:
    No actuators configured
        --> No actuator VAEs are required.
    Actuators configured
        --> All actuator signals must have corresponding actuator VAEs.

Output signals:
    Outputs configured without VAEs
        --> Output signals are kept in real space (no compression).
    Outputs configured with VAEs
        --> All output signals must have corresponding output VAEs.

Returns
-------
Saved model and loss curves

author: andrea.loreti@ukaea.uk
"""
import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
import yaml
from typing import List

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tokamark.tasks import get_task_metadata
from tokamark.data import initialize_TokaMark_dataset

from src.utils.utils import (read_data_split_csv, ComposeTransforms, get_train_test_val_shots, initialize_datasets)
from src.benchmark.utils import load_task_config, load_benchmark_settings, parse_args, load_vae_model, create_vae_dictionary
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.vae_pipeline.models.vae_model import beta_VAE
from src.utils.layer_factory import SequentialBuilder
from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform, ReplaceNaN
from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform

def process_data(
    models: List[beta_VAE], 
    batched_data: List[torch.Tensor]):
    """
    - If model is not None
        For each pair of batched_data and model we get the latent space representation of the 
        data by calling the model encoder. 

    - If model is None
        For batched_data without a paired model data are reshaped to ndim = 2.

    Assumes:
    1-  Batch_data and models are correctly alligned. 
        For instance, the first entry in models is associated with the first entry in batched_data. 

    2-  The number of models and entries in batched_data must be the same. 

    Parameters
    ----------
    models :  list[beta_VAE] or None
        List of pre-trained beta_VAE models or None entries.
    batched_data : list[torch.Tensor]
        List of batched tensors in the real space, each tensor has at least 2 dimensions.
   
    Returns
    -------
    data : list[torch.Tensor]
        List of batched tensors after processing. 
        The processing consits in two mutually exclusive options:

            1) The tensors in the batch are compressed in the latent space representation, if
            corresponding VAEs exist in models.

            2) The tensors in the batch are kept in their representations in the real space 
            but reshaped to agree with the dimesnion of compressed tensors, i.,e., 2D

    """

    # Sanity check
    if len(models) != len(batched_data):
        raise ValueError(
            f"Allignment problem: \
            the length of the list of models: ({len(models)}), is different \
            from the number of data given in the list of data: ({len(batched_data)})"
        )

    # Return values
    data: List[torch.Tensor] = []   # One process data (entry) for each signal in batch_data 
    
    # Encode or reshape tensors
    for model, batch in zip(models, batched_data):
        
        # Check nr. of dimensions
        if batch.ndim < 2:
            raise ValueError("Each batch must have at least 3 dimensions [B, C, L]")

        # Check data for encoding
        if model is not None:
        
            p = next(model.parameters())
            if batch.device != p.device or batch.dtype != p.dtype:
                raise ValueError(f"Batch is not on the same device as the model: {p.device}")

            with torch.no_grad():
                z = model.encode(batch)[0]
  
        else:
            z = batch.reshape(batch.shape[0], -1)

        # Ensure output is 2D [B, latent_dim]
        if z.ndim != 2 or z.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Encoder output shape mismatch: expected [B, D], got {z.shape}"
            )

        data.append(z)
        
    return data

    
def process_batch(
        batch,
        vae_dictionary,
        verbose = True
    ):
    

    """
    Preprocess a batch by aligning dtypes/devices, extracting VAE latent representations,
    processing signals which are not encoded into the latent space.

    Parameters
    ----------
    batch : dict
        A batch dictionary with at least:
        - `batch['x']`: list[torch.Tensor] to contain the input tensors plus actuators,
        - `batch['y']`: list[torch.Tensor] to contain the output tensors,
        Tensors are expected to share the same batch size `B` on shape[0]. 
        
    vae_dictionary : dict[str, beta_VAE]
         vae_dictionary = {
            "input": dict[str, beta_VAE],
            "actuator": dict[str, beta_VAE] or [], 
            "output": dict[str, beta_VAE] or []
         }
    sentinel_value : float
        A place holder (99.0) that replaced NaN

    Returns
    -------
    input_data : torch.Tensor
        Concatenated latent representation for inputs/actuators.
    target_data : torch.Tensor
        Concatenated latent representation for outputs/targets.
    """

    # Original data
    x = batch['x'] # Input + actuator
    y = batch['y'] # Output

    # Alignment and data transfer
    try:
        first_input_model = next(iter(vae_dictionary["input"].values()))
        p = next(first_input_model.parameters())
        input_  = [x_.to(dtype = p.dtype, device = p.device) for x_ in x] # real space data
        target_ = [y_.to(dtype = p.dtype, device = p.device) for y_ in y] # real space target
    except Exception as e:
        raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")

    # When the signals are absent from the shot the MAST_tools sets them to NaN
    for i, inp in enumerate(input_):
        input_[i] = torch.nan_to_num(input_[i], nan=0, posinf=0, neginf=0)
    for j, tar in enumerate(target_):
        target_[j] = torch.nan_to_num(target_[j], nan=0, posinf=0, neginf=0)

    # Collect VAEs
    input_vae = list(vae_dictionary["input"].values()) + \
                list(vae_dictionary["actuator"].values())
    target_vae = list(vae_dictionary['output'].values())

    # Inputs are always encoded, hence encode_masks is used, see process_data method.
    input_data_list = process_data(input_vae, input_) 

    # Targets can be either encoded or left in the real space in either cases we use the masks return 
    # value of process_data since this offers a mask per entry of each tensor.
    target_data_list = process_data(target_vae, target_)

    input_data = torch.cat(input_data_list, dim=1)
    target_data = torch.cat(target_data_list, dim=1)

    return input_data, target_data
    
def loss_function(reco, target, eps = 1e-8):
    """
    Compute a mean squared error loss using an explicit validity mask.

    The loss is computed per sample over valid target entries only
    and then averaged over the batch.

    Args:
        reco (torch.Tensor): Reconstructed output tensor, shape [B, ...].
        target (torch.Tensor): Target tensor, shape [B, ...].
        mask (torch.Tensor): tensor, same shape as target, 1 (0) valid (invalid) entries.
        weights : (torch.Tensor) 
            for each sampe b in B, a weight is given that provides % of completness.
        eps (float, optional): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if target.ndim !=2:
        print(f"WARNING target ndim: expected 2 but got {target.ndim}")

    # Reduce over all non-batch dimensions
    dims = tuple(range(1, target.ndim))

    squared_diff = (target - reco)**2 # [B,L]
    loss_per_sample = squared_diff.mean(dim=dims) # [B]

    # Compute weighted mean loss per batch only on valid samples
    loss = loss_per_sample.mean()

    return loss
    
def train_model(
    SETTINGS:SettingsBenchmark,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    vae_dictionary,
    model,
    optimizer,
    scheduler,
    device,
    output_directory,
    use_amp,
    grad_clip,
    verbose = True
    ):

    train_vs_epoch = []
    val_vs_epoch = []
    
    best_val_loss = float("inf")
    epochs_no_improvement = 0
    
    # Loop through epochs
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    stop_early = False
    
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        
        if verbose:
            print(f"Epoch {epoch}")

        if stop_early:
            break
        
        # Initialize 
        train_loss = 0.0
        train_counts =  0
        val_loss = 0.0
        val_counts = 0

        # TRAINING
        model.train()
        
        # Loop thrpough batches
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 100 == 0:
                if verbose:
                    print(f"\nBatch {batch_idx}")
            
            data, target = process_batch(
                batch,
                vae_dictionary,
                verbose = False)
            
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    
                    reconstruction = model(data)

                    loss = loss_function(reconstruction, target)
                    
                    if (not torch.isfinite(loss)):
                        if verbose:
                            print(
                                f"[Training batch {batch_idx} non-finite loss components "
                                f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                            )
                        continue 

                    scaler.scale(loss).backward()
            else:
                reconstruction = model(data)
                loss = loss_function(reconstruction, target)
                if (not torch.isfinite(loss)):
                    if verbose:
                        print(
                            f"[Training batch {batch_idx} non-finite loss components "
                            f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                        )
                    continue  
                loss.backward()
            
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
                
            # Initialize, Propagate, Step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Accumulate
            train_loss += loss.item()
            train_counts += 1
        
        train_vs_epoch.append(train_loss/max(1,train_counts))
            
        scheduler.step()  
        
        # EVALUATION
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                
                if batch_idx % 100 == 0:
                    if verbose:
                        print(f"\nBatch {batch_idx}")

                data, target = process_batch(
                    batch,
                    vae_dictionary,
                    verbose = False)
                
                if data is None:
                    continue
     
                if use_amp:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        reconstruction = model(data)
                        loss = loss_function(reconstruction, target, valid_target, weights)
                else:
                    reconstruction = model(data)
                    loss = loss_function(reconstruction, target, valid_target, weights)
                
                if (not torch.isfinite(loss)):
                    if verbose:
                        print(
                        f"[batch {batch_idx} non-finite loss components "
                        f"loss finite={torch.isfinite(loss)}; skipping sub-batch."
                        )
                    continue 
                
                val_loss += loss.item()
                val_counts += 1
        
        avg_val_loss = val_loss / max(1,val_counts)
        val_vs_epoch.append(avg_val_loss)
        
        # Save best model
        if  best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improvement = 0
            
            # Save best model state
            model_path = os.path.join(output_directory, f"best_model.pt")
            
            print(f"BEST LOSS FOUND, epoch {epoch}")
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

        if verbose:         
            print(f"'train_losses': {train_vs_epoch}")
            print(f"'val_losses': {val_vs_epoch}")
            
        with open(os.path.join(output_directory, 'loss_curves.json'), 'w') as f:
            data = {
                'train_losses': train_vs_epoch,
                'val_losses': val_vs_epoch
            }
            json.dump(data, f, indent=4)
            
def main():    
    
    # Determine device to train on
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--------------- RUNNING ON GPUs ---------------")
    else:
        device = torch.device("cpu")
        print(f"--------------- RUNNING ON CPUs ---------------")
        
    args = parse_args()

    config_task_file_path: str = args.config_task_file_path
    config_benchmark_file_path: str = args.config_benchmark_file_path
    config_benchmark_file_name: str = os.path.basename(config_benchmark_file_path)
    
    print(f"config_task_file_path = {config_task_file_path}")
    print(f"config_benchmark_file_path = {args.config_benchmark_file_path}")
    
    # Load task config
    try:
        config_task = load_task_config(config_task_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Load model settings
    try:
        SETTINGS = load_benchmark_settings(config_benchmark_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    output_directory = SETTINGS.LOCAL_PATHS.output_directory + config_benchmark_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
    
    # Get lists of shot IDs for train, test and val samples
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = None,
        csv_path = SETTINGS.LOCAL_PATHS.data_split_csv_path
    )
    
    # Initialize task specific metadata
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )
    
    # Get source-signal
    source_signal_list = (
        (config_task["sources_and_signals"].get("input_name") or [])
        + (config_task["sources_and_signals"].get("actuator_name") or [])
        + (config_task["sources_and_signals"].get("output_name") or [])
    )
    # Uniqueness
    source_signal_list = [
        s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]
    ]    
        
    # Open file containing mean and std values
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_signals_stats.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)

    # Signal-level transform map. It is common to all signals whether inputs, actuators or targets.
    sentinel_value = 99.0
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                ImputerTransform()
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }
    
    # MAST base datasets
    zarr_local_path = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2"
    store_mast_settings = {"base_local_zarr_path":zarr_local_path} if SETTINGS.local and zarr_local_path else None
    base_datasets = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": []},
        signal_transform_map=signal_transform_map,
        shot_transforms={},
        local_flag=SETTINGS.local,
        cache_data=False,
        return_incomplete_shots = False,
        store_mast_settings=store_mast_settings
    )

    base_train_dataset = base_datasets['train']
    base_val_dataset = base_datasets['val']
    
    model_specific_transform = ModelSpecificTransform()
    
    # Specific datasets
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

    # DataLoaders
    train_dataloader = DataLoader(
        dataset = train_model_dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =  SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )
    val_dataloader = DataLoader(
        dataset = val_model_dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =  SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )

    # Load VAEs 
    # ----------------------------------------
    vae_dictionary = {"input": {}, "actuator": {}, "output": {}}
    create_vae_dictionary(vae_dictionary, "input", config_task["sources_and_signals"].get("input_name"), SETTINGS, SETTINGS.LOCAL_PATHS.input_vae_models)
    create_vae_dictionary(vae_dictionary, "actuator", config_task["sources_and_signals"].get("actuator_name"), SETTINGS, SETTINGS.LOCAL_PATHS.actuator_vae_models)
    create_vae_dictionary(vae_dictionary, "output", config_task["sources_and_signals"].get("output_name"), SETTINGS, SETTINGS.LOCAL_PATHS.output_vae_models)

    # Move VAEs to device:
    for group in ("input", "actuator", "output"):
        for m in vae_dictionary[group].values():
            if m is None:
                continue
            m.to(device)
            m.eval() 

    # Retrieve model architecture
    if SETTINGS.MODEL.layers is None:
        raise ValueError("Model layers not specified correctly in task config .json")
    
    # Initialize model and send it to device
    model = SequentialBuilder({"layers": SETTINGS.MODEL.layers})
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = SETTINGS.TRAINING.lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
             optimizer,
             T_0 = SETTINGS.TRAINING.num_epochs,
             T_mult = 1, 
             eta_min = int(SETTINGS.TRAINING.lr/10)
            )
    
    ########### Use this block to continue training from a specific checkpoint ####
    # model_path = "src/benchmark/data/output/task1_1_config_v3_part1/best_model.pt"
    # print(f"RESUMING TRAINING from {model_path}")
    # checkpoint = torch.load(model_path, map_location='cuda')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to('cuda')
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    with open(os.path.join(output_directory, "model.json"),'w') as f:
       json.dump(
            str(model),
            f,
            indent=4
            )
    
    # Save config file 
    try:
        with open(config_benchmark_file_path, 'rb') as src, open(os.path.join(output_directory,config_benchmark_file_name), 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Error copying config file: {e}")
        
    # Start training/validating
    train_model(
        SETTINGS,
        train_dataloader,
        val_dataloader,
        vae_dictionary,
        model,
        optimizer,
        scheduler,
        device,
        output_directory,
        use_amp = True,
        grad_clip=1,
        verbose = True
        )
    
if __name__ == "__main__":
    main()
