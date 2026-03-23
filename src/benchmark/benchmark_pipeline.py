import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
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

sys.path.insert(0, "fairmast_data_processing/src")
from fairmast_data_processing.src.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from fairmast_data_processing.src.MAST_benchmark.tools.transforms.stdscale_transform import StdScalingTransform
from fairmast_data_processing.src.MAST_benchmark.tasks import get_task_metadata
from fairmast_data_processing.src.MAST_benchmark.data import initialize_MAST_dataset
from fairmast_data_processing.src.MAST_benchmark.data import (initialize_TokaMark_dataset)
from fairmast_data_processing.scripts.test_pipeline import ModelSpecificTransform
                                          
from src.vae_pipeline.utils.utils import (read_data_split_csv, ComposeTransforms)
from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.vae_pipeline.utils.utils import get_train_test_val_shots
from src.benchmark.utils import load_task_config, load_benchmark_settings, parse_args, load_vae_model
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.vae_pipeline import initialize_datasets
from src.vae_pipeline.utils.layer_factory import SequentialBuilder



def batch_preprocess(
        batch,
        vae_input_models: list[beta_VAE],
        vae_actuator_models: list[beta_VAE],
        vae_output_models: list[beta_VAE],
        device
    ):
    

    """
    Preprocess a mini-batch by aligning dtypes/devices, extracting VAE latent representations,
    and concatenating representations into different tensors for inputs and targets, respectively.

    This function expects:
      - `batch['x']` to contain the input tensors plus actuators,
      - `batch['y']` to contain the output tensors,
      - lists of VAE models for inputs/actuators and outputs,

    Parameters
    ----------
    batch : dict
        A batch dictionary with at least:
          - `batch['x']`: list[torch.Tensor] 
          - `batch['y']`: list[torch.Tensor] 
        Tensors are expected to share the same batch size `B`. Extra trailing dimensions beyond
        feature dims are allowed only if the corresponding VAEs accept them in `encode`.
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
    try:
        p = next(vae_input_models[0].parameters())
    except Exception as e:
        print(f"Error in determining the dtype of your model: {e}")
        return None
    
    data = [x_.to(dtype=p.dtype, device=p.device) for x_ in x]
    target = [y_.to(dtype=p.dtype, device=p.device) for y_ in y]

    # Collect VAEs
    data_vae = [*(vae_input_models or []), *(vae_actuator_models or [])]
    target_vae = [*(vae_output_models or [])]
        
    # Latent space representations
    data_representation = data
    if data_vae:
        try:
            data_representation = [m.encode(d)[0] for m, d in zip(data_vae, data)]
        except Exception as e:
            raise RuntimeError("tensor-model mismatch during `encode` call") from e
                
    target_representation = target
    if target_vae:
        try:
            target_representation = [m.encode(d)[0] for m, d in zip(target_vae, target)]
        except Exception as e:
            raise RuntimeError("tensor-model mismatch during `encode` call") from e
    else:
        target_representation = [t.reshape(t.shape[0], -1) for t in target] # Reshape to (B,N)

    # Concatenate tensors for training, behaviour not tested for tensors with dim>2
    input_data = target_data = None
    if all(d.ndim == 2 for d in data_representation):
        input_data = torch.cat(data_representation, dim=1)
    else:
        input_data = torch.cat([d.flatten(start_dim=1) for d in data_representation], dim=1)
        
    if all(d.ndim == 2 for d in target_representation):
        target_data = torch.cat(target_representation, dim=1)
    else:
        target_data = torch.cat([d.flatten(start_dim=1) for d in target_representation], dim=1)
        
    return input_data, target_data
    
    
    
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

            data, target = batch_preprocess(
                batch,
                vae_input_models, 
                vae_actuator_models,
                vae_output_models,
                device)
            
            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    reconstruction = model(data)
                    loss = F.mse_loss(reconstruction, target, reduction='mean')
                    if (not torch.isfinite(loss).all()):
                        print(
                            f"[Training batch {batch_idx} non-finite loss components "
                            f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                        )
                        continue 
                    scaler.scale(loss).backward()
            else:
                reconstruction = model(data)
                loss = F.mse_loss(reconstruction, target, reduction='mean') 
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

                data, target = batch_preprocess(
                    batch,
                    vae_input_models, 
                    vae_actuator_models,
                    vae_output_models,
                    device)
                
                if use_amp:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        reconstruction = model(data)
                        loss = F.mse_loss(reconstruction, target, reduction='mean') 
                else:
                    reconstruction = model(data)
                    loss = F.mse_loss(reconstruction, target, reduction='mean') 
                
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
    config_model_file_path: str = args.config_model_file_path
    config_model_file_name: str = os.path.basename(config_model_file_path)
    
    print(f"config_task_file_path = {config_task_file_path}")
    print(f"config_model_file_path = {args.config_model_file_path}")
    
    # Load task config
    try:
        config_task = load_task_config(config_task_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Load model settings
    try:
        SETTINGS: SettingsBenchmark = load_benchmark_settings(config_model_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    output_directory = SETTINGS.LOCAL_PATHS.output_directory + config_model_file_name.removesuffix(".json") + "/"
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
                StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                ImputerTransform()
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
        return_incomplete_shots = False
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
    
    # Optionally wrap with cache
    # if SETTINGS.cache:
    #     train_dataset = CachedDataset(train_model_dataset)
    #     val_dataset   = CachedDataset(val_model_dataset)
    # else:
    #     train_dataset = train_model_dataset
    #     val_dataset   = val_model_dataset

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
        print(f"vae paths for inut signals not specified in {config_task_file_path}")
    

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
    model_path = "src/benchmark/data/output/task1_1_config_v3_part1/best_model.pt"
    print(f"RESUMING TRAINING from {model_path}")
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    with open(os.path.join(output_directory, "model.json"),'w') as f:
       json.dump(
            str(model),
            f,
            indent=4
            )
    
    # Save config file 
    try:
        with open(config_model_file_path, 'rb') as src, open(os.path.join(output_directory,config_model_file_name), 'wb') as dst:
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
