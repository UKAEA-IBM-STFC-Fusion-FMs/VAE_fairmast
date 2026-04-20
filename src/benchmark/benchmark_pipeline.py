""" 
    PyTorch pipeline to evaluate trained VAEs over tasks defined in fairmast_data_process.src.benchmark.
    For more details on the benchmark study see arXiv:2602.10132 

    RUN:
    python src/benchmark/benchmark_pipeline.py --config_benchmark_file_path path_to_json_benchmark_file --config_task_file_path fairmast_data_processing/src/MAST_benchmark/tasks_configs/.yaml


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

sys.path.insert(0, "fairmast_data_processing/src")
from fairmast_data_processing.src.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from fairmast_data_processing.src.MAST_benchmark.tools.transforms.stdscale_transform import StdScalingTransform
from fairmast_data_processing.src.MAST_benchmark.tasks import get_task_metadata
from fairmast_data_processing.src.MAST_benchmark.data import initialize_TokaMark_dataset
from fairmast_data_processing.scripts.test_pipeline import ModelSpecificTransform
                                          
from src.vae_pipeline.utils.utils import (read_data_split_csv, ComposeTransforms)
from src.vae_pipeline.utils.utils import get_train_test_val_shots
from src.benchmark.utils import load_task_config, load_benchmark_settings, parse_args, load_vae_model
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.vae_pipeline import initialize_datasets
from src.vae_pipeline.utils.layer_factory import SequentialBuilder


def get_latent_representation(models: List[beta_VAE], batched_real_data: List[torch.Tensor]):
    """Move data from real space to the latent space for those signals which have corresponding VAE.

    Assumes: 
    1-That model.encode(data)[0] returns mu, i.e., latent space repres.
    2-That batch.dtype is float (or isnan(batch) would not work).
    3-That model and batch are on same device.
    4-That the first len(models) tensors in batched_real_data have corresponding VAEs,
    any remaining tensors at the end are kept uncompressed.

    Parameters
    ----------
    models :  list[beta_VAE]
        List of pre-trained beta_VAE models.
    batched_real_data : list[torch.Tensor]
        List of batched of tensors in the real space, each tensor has at least 2D.

    Returns
    -------
    batched_latent_data : list[torch.Tensor]
        List of batched tensors in the latent space representation, 2D shape [B, d_latent].
        Each entry has same batch dimension but different second dimension given by the size of their
        latent space representation.

    masks : list[torch.Tensor]

        A list containing masks where True indicates samples containing at least one NaN 
        in batched_real_data that are going through the encoder.
        Each entry has shape [B].
  
    """
    # Sanity check
    if len(models) > len(batched_real_data):
        raise ValueError(
            f"More models ({len(models)}) than data tensors ({len(batched_real_data)})"
        )

    # Return values
    batched_latent_data: List[torch.Tensor] = []
    masks = []

    # Encode tensors that have a corresponding model
    for model, batch in zip(models, batched_real_data):
        if batch.ndim < 2:
            raise ValueError("Each batch must have at least 2 dimensions [B, ...]")

        dims = tuple(range(1, batch.ndim))
        nan_mask = torch.isnan(batch).any(dim=dims)  # each tensor shape: [B]
        masks.append(nan_mask)

        if nan_mask.any():
            batch_in = batch.clone()
            batch_in[nan_mask] = 0.0
        else:
            batch_in = batch

        try:
            z = model.encode(batch_in)[0]  # expected shape: [B, latent_dim]
        except Exception as e:
            raise RuntimeError(f"tensor-model mismatch during `encode`: {e}")

        # Ensure output is 2D [B, latent_dim]
        if z.ndim != 2 or z.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Encoder output shape mismatch: expected [B, D], got {z.shape}"
            )

        batched_latent_data.append(z)

    return batched_latent_data, masks

def process_extra_tensors(
    models, 
    batched_real_data, 
    batched_latent_data,
    masks, 
    impute_with_zeros):
    """
    
    Update list of tensors in the latent space representation 
    with re-shaped tensors (to match the latent space representaion shape), of data that had not been comprtessed
    with VAE.

    Update masks of NaN entries in the list.
    
    Parameters
    ----------
    models :  list[beta_VAE]
        List of pre-trained beta_VAE models.
    batched_real_data : list[torch.Tensor]
        List of batched tensors in the real space, each tensor has at least 2D.
    batched_latent_data : list[torch.Tensor]
        List of batched tensors in the latent space representation. Itslength is len(models).
        Each entry has same batch dimension but different second dimension.
        This list will be updated and returend.
    masks : list[torch.Tensor]
        A list of length = len(models) containing masks (True) for original NaN 
        in batched_real_data that have been compressed.
        Each entry has shape [B]. 
        This list will be updated and returned
    impute_with_zeros : bool
        Choose whether or not replace NaN with zeros.
        If impute_with_zeros=False, NaNs are propagated to the output

    Raises
    ------
    ValueError

    """
    # Process remaining tensors without models
    for batch in batched_real_data[len(models):]:

        nan_mask = torch.isnan(batch) 
        masks.append(nan_mask.reshape(batch.shape[0], -1))

        if nan_mask.any() and impute_with_zeros:
            batch_in = batch.clone()
            batch_in[nan_mask] = 0.0
        else:
            batch_in = batch

        batched_latent_data.append(batch_in.reshape(batch_in.shape[0], -1))
    
    return batched_latent_data, masks
    
def batch_preprocess(
        batch,
        vae_input_models: list[beta_VAE],
        vae_actuator_models: list[beta_VAE],
        vae_output_models: list[beta_VAE]
    ):
    

    """
    Preprocess a batch by aligning dtypes/devices, extracting VAE latent representations,
    processing signals which do not need to be encoded into the latent space,
    concatenating representations into different tensors for inputs and targets, respectively.

    Input tensors are returned together with masks that are True for NaN in the original data.

    This function expects:
    - `batch['x']` to contain the input tensors plus actuators,
    - `batch['y']` to contain the output tensors,
    - lists of VAE models for inputs/actuators and outputs,

    Assumes all VAEs are on same device and dtype.

    Parameters
    ----------
    batch : dict
        A batch dictionary with at least:
        - `batch['x']`: list[torch.Tensor] 
        - `batch['y']`: list[torch.Tensor] 
        Tensors are expected to share the same batch size `B` on shape[0]. 
    vae_input_models : list[beta_VAE]
        List of VAE models used to encode the input signals. Must be non-empty because this function
        uses `vae_input_models[0]` to determine the reference dtype/device.
    vae_actuator_models : list[beta_VAE]
        Optional list of VAE models to encode actuator signals;
    vae_output_models : list[beta_VAE]
        Optional/ list of VAE models to encode output signals.
    device:
        current device, either 'cpu' or 'cuda'

    Returns
    -------
    input_data : torch.Tensor
        Concatenated latent representation for inputs/actuators.
        Concatenated latent representation for outputs/targets.

    """

    # Original data
    x = batch['x'] # Input + actuator
    y = batch['y'] # Output
    
    # Match data-model types
    if not vae_input_models:
        raise ValueError("Input VAEs must not be empty")

    try:
        # Find device and send data
        p = next(vae_input_models[0].parameters())
        device_type = p.dtype
        data = [x_.to(dtype=device_type, device=p.device) for x_ in x] # real space data
        target = [y_.to(dtype=device_type, device=p.device) for y_ in y] # real space target
    except Exception as e:
        raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")
 
    # Collect VAEs
    data_vae = [*(vae_input_models or []), *(vae_actuator_models or [])]
    target_vae = [*(vae_output_models or [])]

    # Prepare input data
    data_representation, mask_data = get_latent_representation(data_vae, data)
    input_data, mask_input = process_extra_tensors(
                                            data_vae, 
                                            data, 
                                            data_representation,
                                            mask_data, 
                                            impute_with_zeros = True)

    
    mask_input_tensor = torch.stack(mask_input, dim=1)
    if not mask_input_tensor.any():
        return None, None
    
    target_representation, mask_target = get_latent_representation(target_vae, target)
    target_data, mask_target  =  process_extra_tensors(
                                            target_vae, 
                                            target, 
                                            target_representation,
                                            mask_target, 
                                            impute_with_zeros = True)

    # Concatenate
    input_data = torch.cat(data_representation, dim=1)
    mask_input = torch.stack(mask_input, dim=1)
    valid_input = ~mask_input # Change logic
    input = torch.cat([input_data,valid_input.to(dtype=input_data.dtype, device=input_data.device)],dim=1)
    
    target_data = torch.cat(target_representation, dim=1)
    mask_target = torch.cat(mask_target, dim=1)
    valid_target = ~mask_target
    
    return input, target_data, valid_input, valid_target
    
    
def masked_loss(reco, target, valid_target, eps = 1e-8):
    """
    Compute a mean squared error loss using an explicit validity mask.

    The loss is computed per sample over valid target entries only
    and then averaged over the batch.

    Args:
        reco (torch.Tensor): Reconstructed output tensor, shape [B, ...].
        target (torch.Tensor): Target tensor, shape [B, ...].
        valid_target (torch.Tensor): Boolean tensor, same shape as target.
                                     True indicates valid entries.
        eps (float, optional): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar loss value.
    """



    if target.shape != valid_target.shape:  
        raise ValueError(
            f"target and valid_target must have the same shape, "
            f"got {target.shape} and {valid_target.shape}"
        )

    # Convert validity mask to float for arithmetic
    mask = valid_target.to(dtype=target.dtype)

    # Reduce over all non-batch dimensions
    dims = tuple(range(1, target.ndim))
    valid_per_sample = mask.sum(dim=dims) # nr. of valid entries per sample 

    squared_diff = mask * (target - reco)**2
    loss_per_sample = squared_diff.sum(dim=dims) # per sample in batch

    mean_loss_per_sample = loss_per_sample/(valid_per_sample + eps) # average loss
    
    # Compute mean loss per batch only on valid samples
    has_valid = valid_per_sample > 0
    if has_valid.any():
        return mean_loss_per_sample[has_valid].mean()
    else:
        return torch.zeros((), device=target.device, dtype=target.dtype)

    
def train_model(
    SETTINGS:SettingsBenchmark,
    train_dataloader:DataLoader,
    val_dataloader:DataLoader,
    vae_input_models: list[beta_VAE],
    vae_actuator_models: list[beta_VAE],
    vae_output_models: list[beta_VAE],
    model,
    optimizer,
    scheduler,
    device,
    output_directory,
    use_amp,
    grad_clip
    ):
    
    train_vs_epoch = []
    val_vs_epoch = []
    
    best_val_loss = float("inf")
    epochs_no_improvement = 0
    
    # Loop through epochs
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    stop_early = False
    
    for epoch in range(SETTINGS.TRAINING.num_epochs):

        print(f"Epoch {epoch}")
        if stop_early:
            break
        
        # Initialize 
        train_loss = 0.0
        train_counts =  0
        val_loss = 0.0
        val_counts = 0

        model.train()
        
        # Loop thrpough batches
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx}")

            data, target, _, valid_target = batch_preprocess(
                batch,
                vae_input_models, 
                vae_actuator_models,
                vae_output_models)

            if data is None:
                continue

            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    reconstruction = model(data)
                    loss = masked_loss(reconstruction, target, valid_target)#F.mse_loss(reconstruction, target, reduction='mean')
                    if (not torch.isfinite(loss).all()):
                        print(
                            f"[Training batch {batch_idx} non-finite loss components "
                            f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                        )
                        continue 
                    scaler.scale(loss).backward()
            else:
                reconstruction = model(data)
                loss = masked_loss(reconstruction, target, valid_target)
                if (not torch.isfinite(loss).all()):
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
                    print(f"\nBatch {batch_idx}")

                data, target, _, valid_target = batch_preprocess(
                    batch,
                    vae_input_models, 
                    vae_actuator_models,
                    vae_output_models)
                
                if data is None:
                    continue

                if use_amp:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        reconstruction = model(data)
                        loss = masked_loss(reconstruction, target, valid_target)
                else:
                    reconstruction = model(data)
                    loss = masked_loss(reconstruction, target, valid_target)
                
                if (not torch.isfinite(loss).all()):
                        print(
                            f"[batch {batch_idx} non-finite loss components "
                            f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
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
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_stats_metadata.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)

    # Signal-level transform map. It is common to all signals whether inputs, actuators or targets.
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std'])
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }
    
    # MAST base datasets
    base_datasets = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": []},
        signal_transform_map=signal_transform_map,
        shot_transforms={},
        local_flag=SETTINGS.local,
        cache_data=False,
        return_incomplete_shots = True
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
    
    # Load input VAEs 
    # ----------------------------------------
    vae_input_models = None
    if SETTINGS.LOCAL_PATHS.input_vae_models:
        for i, (source, signal_name) in enumerate(config_task["sources_and_signals"].get("input_name")):
            if signal_name not in SETTINGS.LOCAL_PATHS.input_vae_models[i]:
                raise ValueError("Rectify order of input_vae_models in confij.json to agree with the order in your task settings")
        
        vae_input_models = [load_vae_model(os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, this_vae)) 
                            for this_vae in SETTINGS.LOCAL_PATHS.input_vae_models]
        
        none_indices = [i for i, v in enumerate(vae_input_models) if v is None]

        if none_indices:
            print(f"No vae found at indices: {none_indices}")
            
    else:
        print(f"vae paths for ipnut signals not specified in {config_task_file_path}")
    

    # Load actuator VAEs 
    # ----------------------------------------
    vae_actuator_models = None
    if SETTINGS.LOCAL_PATHS.actuator_vae_models:
        for i, (source, signal_name) in enumerate(config_task["sources_and_signals"].get("actuator_name")):
            if signal_name not in SETTINGS.LOCAL_PATHS.actuator_vae_models[i]:
                raise ValueError("Rectify order of actuator_vae_models in confij.json to agree with the order in your task settings")
            
        vae_actuator_models = [load_vae_model(os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, this_vae)) 
                               for this_vae in SETTINGS.LOCAL_PATHS.actuator_vae_models]
    
    
    # Load output VAEs 
    # ----------------------------------------
    vae_output_models = None
    if SETTINGS.LOCAL_PATHS.output_vae_models:
        for i, (source, signal_name) in enumerate(config_task["sources_and_signals"].get("output_name")):
            if signal_name not in SETTINGS.LOCAL_PATHS.output_vae_models[i]:
                raise ValueError("Rectify order of output_vae_models in confij.json to agree with the order in your task settings")
        
        vae_output_models = [load_vae_model(SETTINGS.LOCAL_PATHS.vae_directory, this_vae) 
                            for this_vae in SETTINGS.LOCAL_PATHS.output_vae_models] 
    
    # Move VAEs to device
    vae_input_models = [model.to(device) for model in vae_input_models]
    if vae_actuator_models:
        vae_actuator_models = [model.to(device) for model in vae_actuator_models]
    if vae_output_models:
        vae_output_models = [model.to(device) for model in vae_output_models]
    
    # Check VAE-data consistency
    if vae_input_models is None:
        raise ValueError("Input signals must all have corresponding VAE. Input VAE list found empty")
    else:
        if len(config_task["sources_and_signals"].get("input_name")) != len(vae_input_models):
            raise ValueEroor(" The number of VAE for input signals is different from the number of input signals")
    
    if config_task["sources_and_signals"].get("actuator_name") is not None:
        if vae_actuator_models is None:
            raise ValueError("Use of actuators without corresponding VAEs NOT ALLOWED. Define VAE models for actuators using the config.json file")
        else:
            if len(config_task["sources_and_signals"].get("actuator_name")) != len(vae_actuator_models):
                raise ValueEroor(" The number of VAE for actuator signals is different from the number of actuator signals")
    
    if vae_output_models is not None:
        if len(config_task["sources_and_signals"].get("output_name")) != len(vae_output_models):
            raise ValueEroor("The number of VAE for output signals is different from the number of output signals")

    
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
        vae_input_models,
        vae_actuator_models,
        vae_output_models,
        model,
        optimizer,
        scheduler,
        device,
        output_directory,
        use_amp = True,
        grad_clip=1
        )
    
if __name__ == "__main__":
    main()
