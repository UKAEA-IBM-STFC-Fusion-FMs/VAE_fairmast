"""
    How to call it, example"
    python src/benchmark/benchmark_evaluation.py --config_benchmark_file_path src/benchmark/configs/task1_3_config.json --config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_1-3.yaml
"""

from typing import Iterable, Optional
import matplotlib.pyplot as plt
import json
import os
import sys
import torch
from torch.utils.data import DataLoader
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

from tokamark.tasks import get_task_metadata
from tokamark.data import initialize_TokaMark_dataset
from tokamark.tools.transforms.reshape_lcfs_transform import  ReshapeLcfsTransform

from src.benchmark.benchmark_model import BenchmarkModel                    
from src.benchmark.utils import (
                                load_benchmark_settings, 
                                parse_args,
                                create_vae_dictionary,
                                process_batch, 
                                masked_loss)
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.benchmark.configs.benchmark_setup import SettingsBenchmark


from src.utils.utils import ( load_task_config, ComposeTransforms, get_train_test_val_shots, initialize_datasets)
from src.utils.layer_factory import SequentialBuilder
from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform, StdDescalingTransform


def hist_rmse(
    rmse:Iterable[float],
    title: str = "RMSE: task nr:",
    xlabel: str = "RMSE",
    save_path: Optional[str] = None) -> None:
    
    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.linspace(min_rmse, max_rmse, 200)

    p95_rmse = float(np.quantile(rmse, 0.95))
    if isinstance(rmse, list):
        mean_rmse = float(np.mean(rmse))
        std_rmse = float(np.std(rmse))
    if isinstance(rmse, torch.Tensor):
        mean_rmse = rmse.mean().item()
        std_rmse = rmse.std().item()
    
    fig, ax =  plt.subplots()
    ax.hist(rmse, bins=bins)
    ax.axvline(p95_rmse, color='red', linestyle='--', linewidth=1.5, label=f'95% threshold: {p95_rmse:.4g}')
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.legend([f'Items: {len(rmse)}', f'95% threshold: {p95_rmse:.4g}'])
    ax.set_title(title)
    plt.tight_layout()
    
    
    stats_text = (
        f"Mean = {mean_rmse:.4g}\n"
        f"Std = {std_rmse:.4g}"
    )

    ax.text(
        0.98, 0.82, stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black",
            alpha=0.8,
        ),
    )


    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)  
    else:
        plt.show()

def get_RMSE(reco: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    RMSE per sample across valid features.
    mask: bool or {0,1} with same shape as reco/target; True/1 means valid.
    """

    if target.ndim != 2:
        raise ValueError(
            f"Target ndim must be 2. Here we have ndim = {target.ndim}"
        )

    if target.shape != mask.shape:  
        raise ValueError(
            f"target and valid_target must have the same shape, "
            f"got {target.shape} and {mask.shape}"
        )
    
    
    dims = tuple(range(1, target.ndim))
    
    valid_entries_per_sample = mask.sum(dim=dims) # shape [B]
    sqr_diff_sum = (mask * (reco - target) ** 2).sum(dim=dims) # shape [B]
    
    mse = sqr_diff_sum / (valid_entries_per_sample + eps) # shape [B]
   
    has_valid = valid_entries_per_sample > 0

    return torch.sqrt(mse[has_valid])

def get_RMSE_for_list_signals(
    reco_list: list[torch.Tensor],
    target_list: list[torch.Tensor],
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    """
    Compute RMSE per sample across valid features for a list of tensors.

    Parameters
    ----------
    reco_list : list[torch.Tensor]
        List of reconstructed tensors. Each tensor must have the same shape
        as the corresponding target and mask tensor.
    target_list : list[torch.Tensor]
        List of target tensors. Each tensor is expected to have shape [B, ...].
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    list[torch.Tensor]
        List of tensors containing RMSE values per sample, after excluding
        samples with no valid entries. For each element, the output shape is
        [n_valid_samples].

    Raises
    ------
    ValueError
        If the input lists do not have the same length, or if corresponding
        tensors do not have matching shapes.
    """
    
    mask_list = [
        torch.isfinite(m).to(dtype=m.dtype, device=m.device)
        for m in target_list
    ]

    
    if not (len(reco_list) == len(target_list) == len(mask_list)):
        raise ValueError(
            f"Input lists must have the same length, got "
            f"{len(reco_list)=}, {len(target_list)=}, {len(mask_list)=}"
        )

    rmse_list = []

    for i, (reco, target, mask) in enumerate(zip(reco_list, target_list, mask_list)):
        if reco.shape != target.shape:
            raise ValueError(
                f"reco and target must have the same shape at index {i}, "
                f"got {reco.shape} and {target.shape}"
            )

        if target.shape != mask.shape:
            raise ValueError(
                f"target and mask must have the same shape at index {i}, "
                f"got {target.shape} and {mask.shape}"
            )

        if target.ndim < 2:
            raise ValueError(
                f"Target ndim must be at least 2 at index {i}. "
                f"Here we have ndim = {target.ndim}"
            )

        dims = tuple(range(1, target.ndim))

        valid_entries_per_sample = mask.sum(dim=dims)                # shape [B]
        sqr_diff_sum = (mask * (reco - target) ** 2).sum(dim=dims)   # shape [B]

        mse = sqr_diff_sum / (valid_entries_per_sample + eps)        # shape [B]
        has_valid = valid_entries_per_sample > 0

        rmse_list.append(torch.sqrt(mse[has_valid]))

    return rmse_list

def decode_reco_signals(output_vaes, signals_latent_space, SETTINGS):
    """Decode signals from latent space representation to real space.

    Args:
        - output_vaes (list[beta_VAE]): contains VAEs for the signals to be decoded.
        - signals_latent_space (batch): contains samples of signals to be decoded. Shape [B,L].
        - SETTINGS: configuration object to retrieve important signal-model parameters.
            1-  SETTINGS.output_signals_len is sued to retrieve every signal length `l` within the sample
                `b` in batch. Sample `b` has total length `L`. This length is the sum_i(l_i).
                For signals without associated VAE, `l` is the length of the signal in the real space 
                once the signal was flattened. For signals with associated VAE, `l` is the length
                of the signal in its latent space representation.
    Returns:
        - signals_real_space List[torch.Tensor]:
        A list of real space (decoded) signals.
    """
    latent_dims = [vae.latent_dim if vae is not None else None for vae in output_vaes]
    
    # decoding individual signals START
    signals_real_space = []
    
    start = 0
    for (i,l) in enumerate(SETTINGS.output_signals_len):
        individual_signal = signals_latent_space[:, start:start + l]
        start = start + l
        
        if latent_dims[i] is not None:
            p = next(output_vaes[i].parameters())
            signals_real_space.append(output_vaes[i].decode(individual_signal.to(dtype=p.dtype, device=p.device))) 
        else: 
            signals_real_space.append(individual_signal)
            
    return  signals_real_space
    
    
def evaluate_model(
    model,
    dataloader: DataLoader,
    vae_dictionary,
    SETTINGS,
    dict_stats_metadata,
    target_names,
    verbose = True
):
    """Evaluate model on a dataloader and compute RMSE.
    """

    # Initialization
    batch_rmse = []
    all_rmse_per_signal = None
    

    # Collect VAEs for inputs, actuators and targets
    actuator_dict = vae_dictionary["actuator"] or {}
    output_dict = vae_dictionary["output"] or {}

    if len(vae_dictionary["input"]) == 0:
        raise ValueError("Input VAE is required.")
    
    input_vae = list(vae_dictionary["input"].values()) + list(actuator_dict.values())
    target_vae = list(output_dict.values())
    

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0 and verbose:
                print(f"\nBatch {batch_idx}")

            data, target, target_mask, target_real_space, _ = process_batch(batch, input_vae, target_vae)

            if data is None:
                continue
            
            reconstruction = model(data)
            loss = masked_loss(reconstruction, target, target_mask)
            
            if not torch.isfinite(loss).item():
                if verbose:
                    print(
                        f"[Training batch {batch_idx} non-finite loss components "
                        f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                    )
                continue  
            
            rmse = get_RMSE(reconstruction,target,target_mask)
            batch_rmse.extend(rmse.tolist())
                
            reco_signals_real_space = decode_reco_signals(target_vae, reconstruction, SETTINGS)
            target_native_space = []
            for i, signal in enumerate(target_real_space):
                reco_signals_real_space[i] = reco_signals_real_space[i].reshape(signal.shape[0],*signal.shape[1:]).to(signal.device)
                name = f"{target_names[i][0]}-{target_names[i][1]}"
                reco_signals_real_space[i]= (reco_signals_real_space[i]  *  dict_stats_metadata[name]["std"]) + dict_stats_metadata[name]["mean"]
                target_native_space.append( (signal *  dict_stats_metadata[name]["std"]) + dict_stats_metadata[name]["mean"])
           


            rmse_list_signals = get_RMSE_for_list_signals(reco_signals_real_space, target_native_space)
            
            if all_rmse_per_signal is None:
                all_rmse_per_signal = [[] for _ in rmse_list_signals]
                
            for i, rmse_signal in enumerate(rmse_list_signals):
                all_rmse_per_signal[i].append(rmse_signal/dict_stats_metadata[name]["std"])
                
    return batch_rmse, all_rmse_per_signal

def main():
    # Determine device to train on
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--------------- RUNNING ON NVIDIA GPU (CUDA) ---------------")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("--------------- RUNNING ON APPLE SILICON GPU (MPS) ---------------")
    else:
        device = torch.device("cpu")
        print("--------------- RUNNING ON CPU ---------------")

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
    _, test_shots, val_shots = get_train_test_val_shots(
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
    signal_transform_map = {}
    all_vars =  [f"{source}-{signal}" for source, signal in source_signal_list]
    for var in all_vars:
        if "lcfs" in var:
            signal_transform_map[var] =  ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                    ReshapeLcfsTransform()
                ]
            )
        else:
            signal_transform_map[var] =  ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[var]['mean'], dict_stats_metadata[var]['std']),
                ]
            )
    
    # MAST base datasets
    zarr_local_path = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2"
    store_mast_settings = {"base_local_zarr_path":zarr_local_path} if SETTINGS.local and zarr_local_path else None
    base_datasets = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": [], "val": val_shots, "test": test_shots},
        signal_transform_map=signal_transform_map,
        local_flag=SETTINGS.local,
        cache_data=False,
        store_mast_settings=store_mast_settings
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
    
    # Load VAEs 
    # ----------------------------------------
    vae_dictionary = {"input": {}, "actuator": {}, "output": {}}
    create_vae_dictionary(device, vae_dictionary, "input", config_task["sources_and_signals"].get("input_name"), SETTINGS, SETTINGS.LOCAL_PATHS.input_vae_models)
    create_vae_dictionary(device, vae_dictionary, "actuator", config_task["sources_and_signals"].get("actuator_name"), SETTINGS, SETTINGS.LOCAL_PATHS.actuator_vae_models)
    create_vae_dictionary(device, vae_dictionary, "output", config_task["sources_and_signals"].get("output_name"), SETTINGS, SETTINGS.LOCAL_PATHS.output_vae_models)

    # Move VAEs to device:
    for group in ("input", "actuator", "output"):
        for m in vae_dictionary[group].values():
            if m is None:
                continue
            m.to(device)
            
    # Initialize model and send it to device
    try:
        model = BenchmarkModel(SETTINGS)
        model.to(device)
    except:
        # Initialize model and send it to device
        print("USING single MLP model as a benchmark model")
        from src.utils.layer_factory import SequentialBuilder
        model = SequentialBuilder({"layers": SETTINGS.MODEL.model_layers})
        model.to(device)
    
    # Initialize model, load parameters and send it to device
    model = BenchmarkModel(SETTINGS)
    

    checkpoint = torch.load(model_path, map_location=device)
    epoch_best_model = checkpoint['epoch']
    print(f"Epoch of the best model: {epoch_best_model}")
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
        

    rmse, all_rmse_per_signal = evaluate_model(
        model,
        test_dataloader,
        vae_dictionary,
        SETTINGS,
        dict_stats_metadata,
        config_task["sources_and_signals"].get("output_name"),
        verbose = True
    )
    
    with open(os.path.join(output_directory, "loss_curves.json"), 'r') as file:
        data = json.load(file)
    
    save_fig_losses_path = os.path.join(output_directory,f"losses_{config_benchmark_file_name.removesuffix('.json')}.pdf")
    

    save_fig_rmse_path = os.path.join(output_directory,f"eval_RMSE_latent_space{config_benchmark_file_name.removesuffix('.json')}.pdf")
    hist_rmse(
        rmse,
        title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
        xlabel = "RMSE",
        save_path = save_fig_rmse_path)

    
    rmse_signals = [torch.cat(t_list, dim=0).cpu() for t_list in all_rmse_per_signal]

    for i,rmse_signal in enumerate(rmse_signals):
        save_fig_loss_path = os.path.join(output_directory,f"eval_RMSE_real_space_{i}_{config_benchmark_file_name.removesuffix('.json')}.pdf")
        hist_rmse(
            rmse_signal,
            title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
            xlabel = "loss",
            save_path = save_fig_loss_path)
    
        
if __name__ == "__main__":
    main()