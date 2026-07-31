"""
python src/benchmark/benchmark_visualization.py --config_benchmark_file_path src/benchmark/configs/task2_1_config_gamma_factor.json --config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_2-1.yaml
"""
from typing import Iterable, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import json
import pickle
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

from src.utils.utils import ( load_task_config, ComposeTransforms, get_train_test_val_shots, initialize_datasets)
from src.utils.layer_factory import SequentialBuilder
from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform


def plot_loss_vs_epoch(
    data: Iterable[float],
    title: str = "Task nr",
    ylabel: str = "Loss",
    xlabel: str = "Epoch",
    save_path: Optional[str] = None) -> None:
        
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

def plot_target_vs_data(worst_reco, best_reco, file_save, x_label, y_label):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cases = [
        ("Best Reconstruction", best_reco, axes[0]),
        ("Worst Reconstruction", worst_reco, axes[1]),
    ]

    for title, reco, ax in cases:
        data = reco[0]
        target = reco[1]
        rmse = reco[2]

        data_np = data.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()

        x = range(len(data_np))

        # Reconstructed signal
        ax.plot(
            x,
            data_np,
            color="blue",
            marker="o",
            markersize=5,
            linewidth=1.5,
            label="Reconstructed"
        )

        # Ground truth
        ax.plot(
            x,
            target_np,
            color="red",
            marker="s",
            markersize=5,
            linewidth=1.5,
            label="Ground Truth"
        )

        ax.set_title(f"{title}\nRMSE = {rmse:.4f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    fig.savefig(file_save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def hist_rmse(
    mse:Iterable[float],
    title: str = "NRMSE: task nr:",
    xlabel: str = "rmse (per window)",
    save_path: Optional[str] = None) -> None:
    
    if isinstance(mse, list):
        RMSE =  np.sqrt(np.mean(mse))
        rmse = np.sqrt(mse)
    if isinstance(mse, torch.Tensor):
        RMSE = torch.sqrt(mse.mean())
        rmse =  torch.sqrt(mse)

    
    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.linspace(min_rmse, max_rmse, 200)

    p95_rmse = float(np.quantile(rmse, 0.95))
    
    fig, ax =  plt.subplots()
    hist = ax.hist(rmse, bins=bins, label=f'Windows: {len(rmse)}')
    ax.axvline(p95_rmse, color='red', linestyle='--', linewidth=1.5, label=f'95% threshold: {p95_rmse:.4g}')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=14, loc="upper right",   bbox_to_anchor=(1.0, 1.0))

    ax.set_title(title)
    plt.tight_layout()
    
    stats_text = (
        f"RMSE = {RMSE:.4g}"
    )

    ax.text(
        0.98, 0.75, stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,  
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

def get_mse(reco: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    MSE per sample across valid features.
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

    return mse[has_valid]

def get_mse_for_list_signals(
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
        as the corresponding target.
    target_list : list[torch.Tensor]
        List of target tensors. Each tensor is expected to have shape [B, ...].
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    list[torch.Tensor]
        List of tensors containing mse values per sample, after excluding
        samples with no valid entries. For each element, the output shape is
        [n_valid_samples].

    Raises
    ------
    ValueError
        If the input lists do not have the same length, or if corresponding
        tensors do not have matching shapes.
    """
    
    valid_entries = [ torch.isfinite(t) for t in target_list]

    if not (len(reco_list) == len(target_list) == len(valid_entries)):
        raise ValueError(
            f"Input lists must have the same length, got "
            f"{len(reco_list)=}, {len(target_list)=}, {len(valid_entries)=}"
        )

    mse_list = []

    for i, (reco, target, valid) in enumerate(zip(reco_list, target_list, valid_entries)):
        

        if reco.shape != target.shape:
            if target.ndim > reco.ndim and target.shape[-1]==1:
                target = target.squeeze(-1)
                valid = valid.squeeze(-1)
            if reco.shape != target.shape:
                raise ValueError(
                    f"reco and target must have the same shape at index {i}, "
                    f"got {reco.shape} and {target.shape}"
                )


        if target.shape != valid.shape:
            raise ValueError(
                f"target and mask must have the same shape at index {i}, "
                f"got {target.shape} and {valid.shape}"
            )

        if target.ndim < 2:
            raise ValueError(
                f"Target ndim must be at least 2 at index {i}. "
                f"Here we have ndim = {target.ndim}"
            )
       
        mask = valid.to(dtype=reco.dtype)
        target[~valid]=0
        
        dims = tuple(range(1, target.ndim))

        valid_entries_per_sample = mask.sum(dim=dims)                # shape [B]
        sqr_diff_sum = (mask * (reco - target) ** 2).sum(dim=dims)   # shape [B]

        mse = sqr_diff_sum / (valid_entries_per_sample + eps)        # shape [B]
        has_valid = valid_entries_per_sample > 0

        mse_list.append(mse[has_valid])

    return mse_list

def decode_reco_signals(output_vaes, signals_latent_space, SETTINGS):
    """Decode signals from latent space representation to real space.

    Args:
        - output_vaes (list[beta_VAE]): contains VAEs for the signals to be decoded.
        - signals_latent_space (batch): contains samples of signals to be decoded. Shape [B,L].
        if previously encoded, i.e., corresponding output_vae exists.
        - SETTINGS: configuration object to retrieve important signal-model parameters.
            1-  SETTINGS.output_signals_len is sued to retrieve every signal length `l` within the sample
                `b` in batch. Sample `b` has total length `L`. This length is the sum_i(l_i).
                For signals without associated VAE, `l` is the length of the signal in the real space 
                once the signal was flattened. For signals with associated VAE, `l` is the length
                of the signal in its latent space representation.
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

def align_shapes(reco_signals_real_space, target_real_space):
    
    for i, (reco, target) in enumerate(zip(reco_signals_real_space,target_real_space)):

        # psi map needs permutation
        if reco.ndim == 4 and reco.shape[1] == target.shape[-1]:
            reco = reco.permute(0, 2, 3, 1)

        # remove trailing singleton dims for lcfs
        while reco.ndim > target.ndim and reco.shape[-1] == 1:
            reco = reco.squeeze(-1)

        while target.ndim > reco.ndim and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # flattened representation
        if reco.shape != target.shape:
            if reco.numel() == target.numel():
                reco = reco.reshape(target.shape)

        assert reco.shape == target.shape, \
            f"Cannot align shapes: {reco.shape} vs {target.shape}"

        reco_signals_real_space[i] = reco.contiguous().to(target.device)
        target_real_space[i] = target

    return reco_signals_real_space, target_real_space
    
def evaluate_model(
    model,
    dataloader: DataLoader,
    vae_dictionary,
    SETTINGS,
    output_directory,
    use_amp = True,
    verbose = True
):
    """Evaluate model on a dataloader and compute losses/RMSE.
    """
    batch_losses = []
    batch_mse = []
    all_mse_per_signal = None
    best_reco = [None, None, float("inf")]
    worst_reco = [None, None, float("-inf")]
    best_rmse =  float("inf")
    worst_rmse = float("-inf")
    
    # Collect VAEs for inputs and targets
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

            data, target, input_mask, target_mask, target_real_space, _ = process_batch(batch, input_vae, target_vae)

            # Exact shape equality
            assert data.shape[0] == target.shape[0] == input_mask.shape[0] == target_mask.shape[0], (
                f"Shapes differ:\n"
                f"data        : {data.shape[0]}\n"
                f"target      : {target.shape[0]}\n"
                f"input_mask  : {input_mask.shape[0]}\n"
                f"target_mask : {target_mask.shape[0]}\n"
            )
            
            for t_real in target_real_space:
                if t_real.shape[0]!=data.shape[0]:
                    raise ValueError(f"target real space batch dimension {t_real.shape[0]} differes from that one of data {data.shape[0]}")
            
            if data is None:
                continue
            
            if use_amp:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    reconstruction = model(data, input_mask)
                    loss = masked_loss(reconstruction, target, target_mask)
            else:
                reconstruction = model(data, input_mask)
                loss = masked_loss(reconstruction, target, target_mask)
            
            if not torch.isfinite(loss).item():
                if verbose:
                    print(
                        f"[Training batch {batch_idx} non-finite loss components "
                        f"loss finite={torch.isfinite(loss).all()}; skipping sub-batch."
                    )
                continue  

            batch_losses.append(loss.item())
            
            mse = get_mse(reconstruction,target,target_mask)
            batch_mse.extend(mse.tolist())
            
            max_value = torch.max(mse)
            max_idx = torch.argmax(mse)
        
            if max_value > worst_reco[2]:
                worst_reco[2] = max_value
                worst_reco[0] = reconstruction[max_idx]
                worst_reco[1] = target[max_idx]
            
            min_value = torch.min(mse)
            min_idx = torch.argmin(mse)
            if min_value < best_reco[2]:
                best_reco[2] = min_value
                best_reco[0] = reconstruction[min_idx]
                best_reco[1] = target[min_idx]
            
            reco_signals_real_space = decode_reco_signals(target_vae, reconstruction, SETTINGS)

            reco_signals_real_space, target_real_space = align_shapes(reco_signals_real_space, target_real_space)
            
            mse_list_signals = get_mse_for_list_signals(reco_signals_real_space, target_real_space)
            mse_per_sample = torch.stack(mse_list_signals, dim=1)
            mse_total = mse_per_sample.sum(dim=1)  
            
            best_idx = torch.argmin(mse_total)
            worst_idx = torch.argmax(mse_total)
     
            if mse_total[best_idx] < best_rmse:
                best_rmse = mse_total[best_idx]
               
                best_reco_real_space = {
                    "sample_idx": int(best_idx),
                    "signals_rec": [x[best_idx].cpu() for x in reco_signals_real_space],
                    "signals_true": [x[best_idx].cpu() for x in target_real_space],
                    "rmse":best_rmse.cpu()
                }

                with open(os.path.join(output_directory,"best_reco.pkl"), "wb") as f:
                    pickle.dump(best_reco_real_space , f)

            if mse_total[worst_idx] > worst_rmse:
                worst_rmse = mse_total[worst_idx]
                worst_reco_real_space = {
                    "sample_idx": int(worst_idx),
                    "signals_rec": [x[worst_idx].cpu() for x in reco_signals_real_space],
                    "signals_true": [x[worst_idx].cpu() for x in target_real_space],
                    "rmse": worst_rmse.cpu()
                }
                with open(os.path.join(output_directory,"worst_reco.pkl"), "wb") as f:
                    pickle.dump(worst_reco_real_space, f)
                    
            if all_mse_per_signal is None:
                all_mse_per_signal = [[] for _ in mse_list_signals]
                
            for i, mse_signal in enumerate(mse_list_signals):
                all_mse_per_signal[i].append(mse_signal)

    return batch_losses, batch_mse, all_mse_per_signal, worst_reco, best_reco, worst_reco_real_space, best_reco_real_space

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
    zarr_local_path = "/lustre/home/bf3280/tokamark_fairmast_dataset"
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
        

    all_losses, rmse, all_mse_per_signal, worst_reco, best_reco, worst_reco_real_space, best_reco_real_space = evaluate_model(
        model,
        test_dataloader,
        vae_dictionary,
        SETTINGS,
        output_directory,
        use_amp = False,
        verbose = True
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
    
    save_fig_rmse_path = os.path.join(output_directory,f"NRMSE_latent_space{config_benchmark_file_name.removesuffix('.json')}.pdf")
    hist_rmse(
        rmse,
        title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
        xlabel = "rmse",
        save_path = save_fig_rmse_path)

    save_fig_loss_path = os.path.join(output_directory,f"All_samples_loss_{config_benchmark_file_name.removesuffix('.json')}.pdf")
    hist_rmse(
        all_losses,
        title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
        xlabel = "loss",
        save_path = save_fig_loss_path)
    
    rmse_signals = [torch.cat(t_list, dim=0).cpu() for t_list in all_mse_per_signal]

    for i,rmse_signal in enumerate(rmse_signals):
        save_fig_loss_path = os.path.join(output_directory,f"NRMSE_real_space_{i}_{config_benchmark_file_name.removesuffix('.json')}.pdf")
        hist_rmse(
            rmse_signal,
            title = f"Task_{config_benchmark_file_name.removesuffix('.json')}",
            xlabel = "rmse",
            save_path = save_fig_loss_path)
    
    save_reco_fig_path = os.path.join(output_directory,f"Reco_examples_{config_benchmark_file_name.removesuffix('.json')}.pdf")
    plot_target_vs_data(worst_reco, best_reco, save_reco_fig_path, "Samples", "Signals")
        
if __name__ == "__main__":
    main()