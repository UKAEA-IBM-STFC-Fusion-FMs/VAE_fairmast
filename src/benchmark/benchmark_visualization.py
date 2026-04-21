from typing import Iterable, Optional, Tuple, Dict
import math
import matplotlib.pyplot as plt
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
import numpy as np

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, "tokamark/src")
from tokamark/src/MAST_tools/MAST_dataset import MastDataset
from tokamark/src/tokamark/tools/transforms/stdscale_transform.pyimport StdScalingTransform
from tokamark.src.tokamark.tasks import get_task_metadata
from tokamark.src.tokamark.data import initialize_TokaMark_dataset
from tokamark.src.scripts.test_pipeline import ModelSpecificTransform

                                          
from src.vae_pipeline.utils.utils import (read_data_split_csv, ComposeTransforms)
from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.vae_pipeline.utils.utils import get_train_test_val_shots
from src.benchmark.utils import load_task_config, load_benchmark_settings, parse_args, load_vae_model
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.vae_pipeline import initialize_datasets
from src.vae_pipeline.utils.layer_factory import SequentialBuilder
from src.benchmark.benchmark_pipeline import batch_preprocess

def _as_2d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D as (N, D) by flattening all non-batch dims."""
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t.flatten(start_dim=1)



def plot_loss_vs_epoch(
    data: Iterable[float],
    title: str = "Task nr",
    ylabel: str = "Loss",
    xlabel: str = "Epoch",
    save_path: Optional[str] = None
) -> None:
        
    val_loss = data["val_losses"]
    train_loss = data["train_losses"]

    # Epochs
    epochs = list(range(1, len(val_loss) + 1))

        
    # Create scatter plot
    fig, ax = plt.subplots()
    ax.plot(epochs, val_loss, linestyle='solid',color='blue', marker='o', label="Validation loss" )
    ax.plot(epochs, train_loss, linestyle='solid',color='red', marker='o', label="Training loss")
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)  
    else:
        plt.show()
    

def hist_rmse(
    rmse:Iterable[float],
    title: str = "RMSE: task nr:",
    xlabel: str = "RMSE",
    save_path: Optional[str] = None
) -> None:
    
    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.linspace(min_rmse, max_rmse, 200)

    p95_rmse = float(np.quantile(rmse, 0.95))
    
    fig, ax =  plt.subplots()
    ax.hist(rmse, bins=bins)
    ax.axvline(p95_rmse, color='red', linestyle='--', linewidth=1.5, label=f'95% threshold: {p95_rmse:.4g}')
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.legend([f'Items: {len(rmse)}', f'95% threshold: {p95_rmse:.4g}'])
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)  
    else:
        plt.show()

 
    
def get_RMSE(reco: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Single overall RMSE across all samples and features."""
    return torch.sqrt(torch.mean((reco - target) ** 2, dim=1))
   

def evaluate_model(
    model,
    dataloader: DataLoader,
    vae_input_models: list[beta_VAE], 
    vae_actuator_models: list[beta_VAE],
    vae_output_models: list[beta_VAE],
    device: Optional[torch.device] = None,
    use_amp = True
):
    """Evaluate model on a dataloader and compute losses/RMSE.
    """
    batch_losses = []
    batch_rmse = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
            
            batch_losses.append(loss.item())
            batch_rmse.extend(get_RMSE(reconstruction,target).tolist())

    return batch_losses, batch_rmse

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
    
    # Load task config
    try:
        config_task = load_task_config(config_task_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Load model settings
    try:
        SETTINGS: SettingsBenchmark = load_benchmark_settings(config_benchmark_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    output_directory = SETTINGS.LOCAL_PATHS.output_directory + config_benchmark_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
    
    model_path = os.path.join(output_directory, "best_model.pt")
    
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
        shots={"train": [], "val": val_shots, "test": test_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms={},
        local_flag=SETTINGS.local,
        cache_data=False,
        return_incomplete_shots = False
    )
    base_val_dataset = base_datasets['val']
    base_test_dataset = base_datasets['test']
    
    model_specific_transform = ModelSpecificTransform()
    
    # Specific datasets
    test_model_dataset = initialize_TokaMark_dataset(
        dataset=base_test_dataset,
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
    test_dataloader = DataLoader(
        dataset = test_model_dataset,
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
            sys.exit(1)
            
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
    
    # Initialize model, load parameters and send it to device
    model = SequentialBuilder({"layers": SETTINGS.MODEL.layers})
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    epoch_best_model = checkpoint['epoch']
    print(f"Epoch of the best model: {epoch_best_model}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    losses, rmse = evaluate_model(
        model,
        val_dataloader,
        vae_input_models, 
        vae_actuator_models,
        vae_output_models,
        device   
    )
    
    with open(os.path.join(output_directory, "loss_curves.json"), 'r') as file:
        data = json.load(file)
    
    save_fig_losses_path = os.path.join(output_directory,f"losses_{config_benchmark_file_name.removesuffix('.json')}.pdf")
    plot_loss_vs_epoch(
    data,
    title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
    ylabel = "Loss",
    xlabel = "Epoch",
    save_path = save_fig_losses_path) 
    
    save_fig_rmse_path = os.path.join(output_directory,f"RMSE_{config_benchmark_file_name.removesuffix('.json')}.pdf")
    hist_rmse(
        rmse,
        title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
        xlabel = "RMSE",
        save_path = save_fig_rmse_path)
    
    
    
    
if __name__ == "__main__":
    main()