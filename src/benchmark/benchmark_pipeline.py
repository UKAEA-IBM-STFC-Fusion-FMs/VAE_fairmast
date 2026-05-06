""" 
PyTorch pipeline to evaluate trained VAEs over tasks defined in tokamark.
For more details on the benchmark study see arXiv:2602.10132 

RUN:
python src/benchmark/benchmark_pipeline.py --config_benchmark_file_path path_to_json_benchmark_file --config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_1-1.yaml


DATA INGESTION:

This pipeline enforces a strict one‑to‑one correspondence between 
configured signals and their associated Variational Autoencoder (VAE) models at ingestion time. 
The configuration defines three categories of signals: inputs, actuators, and outputs, each of which may require compression via a dedicated VAE.

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

def process_data(
    models: List[beta_VAE], 
    batched_data: List[torch.Tensor],
    sentinel_value):
    """
    - If model is not None
        For each pair of batched_data and model we get the latent space representation of the 
        data by calling the model encoder. 

    - If model is None
        For batched_data without a paired model, data are reshaped to ndim = 2.

    Assumes:
    1-  Batch_data and models are correctly alligned. 
        For instance, the first entry in models is associated with the first entry in batched_data. 

    2-  The number of models and entries in batched_data must be the same. 

    Parameters
    ----------
    models :  list[beta_VAE]
        List of pre-trained beta_VAE models or None entries.
    batched_data : list[torch.Tensor]
        List of batched tensors in the real space, each tensor has at least 2 dimensions.
    sentinel_value : same type as data in batch
        This is a placeholder for NaN entries. Its value is 99, outside data distribution.

    Returns
    -------
    data : list[torch.Tensor]
        List of batched tensors after processing. 
        The processing consits in two mutually exclusive options:

            1) The tensors in the batch are compressed in the latent space representation, if
            corresponding VAEs exist in models.

            2) The tensors in the batch are kept in their representations in the real space 
            but reshaped to agree with the dimesnion of compressed tensors, i.,e., 2D

    masks : list[torch.Tensor]
        - If model is not None:
        mask[b, :] = 1 iff sample b had nr of sentinels smaller than a certain cut-off else zeros.
        mask is 1 for all latent features of good samples, 0 otherwise.
        
        - If model is not None:
        mask[b, j] = 1 iff that specific flattened entry is not sentinel.
        mask is 1 per-entry in real space representation.

    encode_masks : list[torch.Tensor]
        1 if signal was encoded and passed cut-off.
        0 if signal did not pass cut-off and was set to zero.
        0 if model does not exist.
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
    masks: List[torch.Tensor] = [] # One mask (entry) for each signal in batch_data 
    encode_masks :  List[torch.Tensor] = []  # One mask (entry) for each signal in batch_data 

    # Encode or reshape tensors
    for model, batch in zip(models, batched_data):
        
        # Check nr. of dimensions
        if batch.ndim < 2:
            raise ValueError("Each batch must have at least 2 dimensions [B, ...]")

        # Batch size
        B = batch.shape[0]

        # Filter invalid samples in the batch based on a cut-off of 0.05 of invalid entries per sample
        dims = tuple(range(1, batch.ndim))
        invalid_entries = torch.isclose(batch, sentinel_value)

        # Initialize mask for samples to encode
        encode_mask = torch.zeros(B, device=batch.device, dtype = batch.dtype)

        # Check data for encoding
        if model is not None:
        
            p = next(model.parameters())
            if batch.device != p.device or batch.dtype != p.dtype:
                raise ValueError(f"Batch is not on the same device as the model: {p.device}")

            if sentinel_value.device != batch.device or sentinel_value.dtype != batch.dtype:
                raise ValueError(f"The 'sentinel_value' is not on the same device or same type as the sample\
                                   device: {sentinel_value.device}, {batch.device},\
                                   types:  {sentinel_value.dtype} {batch.dtype}")

            # Initialize to zero
            z = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            mask = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            
            invalid_fraction_per_sample = invalid_entries.float().mean(dim=dims)  # [B]
            tau = 0.05 
            valid_samples = invalid_fraction_per_sample <= tau  # bool [B]

            if valid_samples.any():
                mask[valid_samples, :]  = 1
                with torch.no_grad():
                    # Clone to avoid corruption of original data
                    x = batch.clone()
                    # Ground invalid entries with zero
                    x[invalid_entries] = 0
                    # Encode valid samples
                    z[valid_samples] = model.encode(x[valid_samples])[0]
                    # Update encode mask
                    encode_mask[valid_samples] = 1
                
        else:
            z = batch.reshape(B, -1).clone()
            mask = (~torch.isclose(z, sentinel_value)).to(dtype=batch.dtype)
            z = z * mask

        # Ensure output is 2D [B, latent_dim]
        if z.ndim != 2 or z.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Encoder output shape mismatch: expected [B, D], got {z.shape}"
            )

        data.append(z)
        masks.append(mask)
        encode_masks.append(encode_mask)
        
    return data, masks, encode_masks

    
def process_batch(
        batch,
        vae_dictionary,
        sentinel_value,
        filtering = 'weights',
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
    input_encode_mask : torch.Tensor
        Tensor of 1 (0) for valid (invalid) entries in `input`.
    valid_target : torch.Tensor
        Tensor of 1 (0) for valid (invalid) entries in `target_data`.
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
    
    sentinel = torch.as_tensor(
                sentinel_value,
                device = input_[0].device,
                dtype = input_[0].dtype
            )

    # When the signals are absent from the shot the MAST_tools sets them to NaN
    # We change them to sentinel_value as for the rest of the analysis.
    for i, inp in enumerate(input_):
        input_[i] = torch.nan_to_num(input_[i], nan=sentinel, posinf=sentinel, neginf=sentinel)
    for j, tar in enumerate(target_):
        target_[j] = torch.nan_to_num(target_[j], nan=sentinel, posinf=sentinel, neginf=sentinel)

    # Collect VAEs
    input_vae = list(vae_dictionary["input"].values()) + \
                list(vae_dictionary["actuator"].values())
    target_vae = list(vae_dictionary['output'].values())

    # Inputs are always encoded, hence encode_masks is used, see process_data method.
    input_data_list, _, input_encode_mask_list = process_data(input_vae, input_, sentinel) 

    # Targets can be either encoded or left in the real space in either cases we use the masks return 
    # value of process_data since this offers a mask per entry of each tensor.
    target_data_list, valid_target_list, _ = process_data(target_vae, target_, sentinel)

    input_data_cat = torch.cat(input_data_list, dim=1)
    target_data = torch.cat(target_data_list, dim=1)

    # if input_data_cat.ndim != 3 or input_encode_mask.ndim != 3 or target_data.ndim !=3:
    #     raise ValueError(f"Input data (mask) dimension expected to be 3, \
    #                     instead is {input_data_cat.ndim} ({input_encode_mask.ndim}) ")
                        
    # Get rid of poor batches
    input_encode_mask = torch.stack(input_encode_mask_list, dim=1)  # [B, n_signals]
    input_sample_completness = input_encode_mask.mean(dim=1)  # [B]
    if (input_sample_completness < 0.75).float().mean().item() > 0.95:
        if verbose:
            print("No valid latent space representation. Too many input signals are missing in this batch. ")
        return None, None, None, None, None
    
    # Concatenate encode mask to signal
    input_data = torch.cat([input_data_cat, input_encode_mask], dim=1)

    # Get rid of poor batches
    valid_target = torch.cat(valid_target_list, dim=1)
    target_sample_completness = (valid_target.mean(dim=1) < 0.75)
    if (target_sample_completness < 0.5).float().mean().item() > 0.50:
        if verbose:
            print("No valid target. Too many output signals are missing in this batch.")
        return None, None, None, None, None
    
    if filtering == 'hard_filtering':
        weights = None
        hard_filter = (input_sample_completness >= 0.75) & (target_sample_completness >= 0.5)

        # Filter data and mask
        input_data = input_data[hard_filter]
        valid_target = valid_target[hard_filter]
        target_data = target_data[hard_filter]
    else:
        weights = torch.minimum(input_sample_completness, target_sample_completness)
        hard_filter = None

    return input_data, target_data, input_encode_mask, valid_target, weights
    
    
def masked_loss(reco, target, mask, weights, eps = 1e-8):
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
    if target.shape != mask.shape:  
        raise ValueError(
            f"target and valid_target must have the same shape, "
            f"got {target.shape} and {mask.shape}"
        )

    if target.ndim !=2:
        print(f"WARNING target ndim: expected 2 but got {target.ndim}")

    # Reduce over all non-batch dimensions
    dims = tuple(range(1, target.ndim))
    valid_per_sample = mask.sum(dim=dims) # nr. of valid entries per sample [B]

    squared_diff = mask * (target - reco)**2 # [B]
    sqr_sum_per_sample = squared_diff.sum(dim=dims) # [B]
    loss_per_sample = sqr_sum_per_sample/(valid_per_sample + eps) # loss per sample # [B]
    
    has_valid = valid_per_sample > 0

    # Compute weighted mean loss per batch only on valid samples
    if weights is not None:
        loss = (weights[has_valid] * loss_per_sample[has_valid]).sum() / weights[has_valid].sum()
    else:
        loss = loss_per_sample[has_valid].mean()

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
    sentinel_value,
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
            
            data, target, input_encode_mask, valid_target, weights = process_batch(
                batch,
                vae_dictionary,
                sentinel_value,
                verbose = False)

            if data is None:
                continue
            
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    
                    reconstruction = model(data)

                    loss = masked_loss(reconstruction, target, valid_target, weights)
                    
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
                loss = masked_loss(reconstruction, target, valid_target, weights)
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

                data, target, input_encode_mask, valid_target, weights = process_batch(
                    batch,
                    vae_dictionary,
                    sentinel_value,
                    verbose = False)
                
                if data is None:
                    continue
     
                if use_amp:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        reconstruction = model(data)
                        loss = masked_loss(reconstruction, target, valid_target, weights)
                else:
                    reconstruction = model(data)
                    loss = masked_loss(reconstruction, target, valid_target, weights)
                
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
                ReplaceNaN(sentinel_value)
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
        sentinel_value = sentinel_value,
        verbose = True
        )
    
if __name__ == "__main__":
    main()
