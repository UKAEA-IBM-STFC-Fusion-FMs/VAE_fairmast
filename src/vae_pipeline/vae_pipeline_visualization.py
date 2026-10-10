"""Plot VAE reconstructions for test windows near a requested physical RMSE.

python src/vae_pipeline/vae_visualization.py --config_file_path src/vae_pipeline/configs/config_equilibrium_psi.json --config_task_file_path src/vae_pipeline/configs/task_encoding_VAE_psi.yaml --output_dir src/vae_pipeline/data/output/conv2d_vae_config_equilibrium_psi
"""

import argparse
import json
import os
import sys
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tokamark.data import initialize_TokaMark_dataset
from tokamark.tasks import get_task_metadata
from tokamark.tools.transforms.reshape_lcfs_transform import ReshapeLcfsTransform

from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform
from src.utils.utils import (
    ComposeTransforms,
    get_train_test_val_shots,
    initialize_datasets,
    load_task_config,
)
from src.utils.plot_signals import plot_psi, plot_loss_vs_epoch
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE


def make_dataset_dataloader(config_file_path: str,config_task_file_path: str):
   
    settings = get_settings(config_file_path)
    config_file_name = os.path.basename(config_file_path)
    source_signal_list = settings.DATA.data_names
    if not source_signal_list or len(source_signal_list) != 1:
        raise ValueError("This utility expects exactly one signal in input.data_names.")

    source, signal_name = source_signal_list[0]
    stats_key = f"{source}-{signal_name}"
    stats_path = os.path.join(
        settings.LOCAL_PATHS.global_mean_std_path, "dict_signals_stats.yaml"
    )
    with open(stats_path, "r", encoding="utf-8") as stats_file:
        stats = yaml.safe_load(stats_file)
    if stats_key not in stats:
        raise KeyError(f"Statistics for {stats_key!r} were not found in {stats_path}.")

    mean = float(stats[stats_key]["mean"])
    std = float(stats[stats_key]["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid mean/std values for {stats_key}: {mean}, {std}.")

    config_task = load_task_config(config_task_file_path)
    task_metadata = get_task_metadata(config_task, verbose=False)
    _, test_shots, _ = get_train_test_val_shots(
        max_index_for_train=settings.TRAINING.num_train_samples,
        max_index_for_val=settings.TRAINING.num_val_samples,
        max_index_for_test=None,
        csv_path=settings.LOCAL_PATHS.data_split_csv_path,
    )

    signal_transform = [StdScalingTransform(mean, std)]
    if "lcfs" in config_file_name.lower():
        signal_transform.append(ReshapeLcfsTransform())
    signal_transform_map = {stats_key: ComposeTransforms(signal_transform)}
    
    local_store = "/lustre/home/bf3280/tokamark_fairmast_dataset"
    store_settings = (
        {"base_local_zarr_path": local_store}
        if settings.DATA.local and local_store
        else None
    )
    base_test_dataset = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": [], "val": [], "test": test_shots},
        signal_transform_map=signal_transform_map,
        local_flag=settings.DATA.local,
        cache_data=False,
        store_mast_settings=store_settings,
    )["test"]
    test_dataset = initialize_TokaMark_dataset(
        dataset=base_test_dataset,
        task_metadata=task_metadata,
        config_metadata=config_task,
        custom_transform=ModelSpecificTransform(settings.WINDOWsSHAPE.permutation),
        test_mode=True,
        shuffle_windows=False,
        verbose=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=settings.TRAINING.dataloader_batch_size,
        num_workers=settings.TRAINING.num_workers,
        persistent_workers=False,
    )

    return settings, mean, std, test_dataset, test_loader

################## RECONSTRUCTIOn METHODS ##################
def reconstruct_window(
    config_file_path: str,
    config_task_file_path: str,
    target_window_index: int,
    target_shot_id: int
    )-> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct signal for a specific `target_window_index` and `target_shot_id`
    """
   
    settings, mean, std, test_dataset, test_loader = make_dataset_dataloader( config_file_path, config_task_file_path)

    source, signal_name = settings.DATA.data_names[0]
    stats_key = f"{source}-{signal_name}"

    training_output_dir = os.path.join(
        settings.LOCAL_PATHS.data_output_directory,
        f"conv2d_vae_{Path(config_file_path).stem}",
    )
    model_path = os.path.join(training_output_dir, f"best_vae_{signal_name}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained VAE checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = beta_VAE(settings)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):

            if batch_index % 100 == 0:
                print(f"Batch {batch_index}")

            mask = ((batch["shot_id"] == target_shot_id) & (batch["window_index"] == target_window_index))
            index = torch.where(mask)[0]
            if len(index) == 0:
                continue

            x = batch["x"][0][index]
           
            parameter = next(model.parameters())
            input_data = x.to(device=device, dtype=parameter.dtype)
            reconstruction, _, _, mask, input_data  = model(input_data, sampling=False)

            if reconstruction.shape != input_data.shape:
                raise ValueError(
                    "Model output and input shapes differ: "
                    f"{tuple(reconstruction.shape)} vs {tuple(input_data.shape)}."
                )
    
            original_physical = input_data * std + mean
            reconstruction_physical = reconstruction * std + mean

    return reconstruction_physical, original_physical

################## STATISTICS METHODS ##################
def rmse_statistics(df)-> dict:
    """
    Compute mean, min and max RMSE and identify the corresponding shot/window. 
    Also find the window whose RMSE is closest to the mean.

    Parameters: pd.DataFrame
            Contains: "shot_id" (int), "window_index" (int), "rmse" (float), "signal_name" (str)
    """

    mean_rmse = float(df["rmse"].mean())

    idx_min = df["rmse"].idxmin()
    idx_max = df["rmse"].idxmax()

    idx_closest_mean = (df["rmse"] - mean_rmse).abs().idxmin()

    min_row = df.loc[idx_min]
    max_row = df.loc[idx_max]
    mean_row = df.loc[idx_closest_mean]

    return {
        "mean_rmse": mean_rmse,
        "min_rmse": {
            "rmse": float(min_row["rmse"]),
            "shot_id": min_row["shot_id"],
            "window_index": min_row["window_index"],
        },
        "max_rmse": {
            "rmse": float(max_row["rmse"]),
            "shot_id": max_row["shot_id"],
            "window_index": max_row["window_index"],
        },
        "closest_to_mean": {
            "rmse": float(mean_row["rmse"]),
            "shot_id": mean_row["shot_id"],
            "window_index": mean_row["window_index"],
            "difference": abs(float(mean_row["rmse"]) - mean_rmse),
        },
    }


def windows_rmse(
    config_file_path: str,
    config_task_file_path: str,
    output_dir: str | None = None) -> pd.DataFrame:
    """
     Build dictionary RMSE containing:
            RMSE["shot_id"]
            RMSE["window_index"]
            RMSE["rmse"]
            RMSE["signal_name"]
        for all shots and windows
    """
   
    settings, mean, std, test_dataset, test_loader = make_dataset_dataloader( config_file_path, config_task_file_path)

    source, signal_name = settings.DATA.data_names[0]
    stats_key = f"{source}-{signal_name}"

    training_output_dir = os.path.join(settings.LOCAL_PATHS.data_output_directory,f"conv2d_vae_{Path(config_file_path).stem}")

    model_path = os.path.join(training_output_dir, f"best_vae_{signal_name}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained VAE checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = beta_VAE(settings)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    RMSE = {"shot_id": [], "window_index": [], "rmse": [], "signal_name":  stats_key}
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):

            if batch_index % 100 == 0:
                print(f"Batch {batch_index}")

            x = batch["x"][0]
            shot_id = batch["shot_id"]
            window_index = batch["window_index"]
           
            parameter = next(model.parameters())
            input_data = x.to(device=device, dtype=parameter.dtype)
            reconstruction, _, _, mask, input_data  = model(input_data, sampling=False)

            if reconstruction.shape != input_data.shape:
                raise ValueError(
                    "Model output and input shapes differ: "
                    f"{tuple(reconstruction.shape)} vs {tuple(input_data.shape)}."
                )

    
            original_physical = input_data * std + mean
            reconstruction_physical = reconstruction * std + mean

            valid = mask.bool()
            if not torch.any(valid & torch.isfinite(reconstruction_physical)):
                raise RuntimeError(
                    "The model produced non-finite reconstruction values at valid "
                    "input positions."
                )

            squared_error = (mask * (reconstruction_physical - original_physical)).square()

            reduce_dims = tuple(range(1, squared_error.ndim))
            valid_count = valid.sum(dim=reduce_dims)
            
            rmse_per_item = torch.sqrt(squared_error.sum(dim=reduce_dims) / valid_count.clamp_min(1))

            RMSE["shot_id"].extend(shot_id.cpu().tolist())
            RMSE["window_index"].extend(window_index.cpu().tolist())
            RMSE["rmse"].extend(rmse_per_item.cpu().tolist())
    
    return pd.DataFrame(RMSE)


################## VISUALIZATION METHODS ##################
def hist_rmse(RMSE, savepath, nr_bins=200):
    """
    Plot histogram of window-level RMSE.

    Parameters
    ----------
    RMSE : dict
        Dictionary containing key 'rmse'.
    nr_bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Existing axis.

    Returns
    -------
    mean_rmse : float
    q95_rmse : float
    """

    rmse = np.asarray(RMSE["rmse"], dtype=float)

    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.linspace(min_rmse, max_rmse, nr_bins)

    mean_rmse = np.mean(rmse)
    q95_rmse = np.quantile(rmse, 0.95)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(rmse, bins=bins, alpha=0.7, color="steelblue")

    ax.axvline(
        mean_rmse,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_rmse:.4f}",
    )

    ax.axvline(
        q95_rmse,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"95% quantile = {q95_rmse:.4f}",
    )

    ax.set_xlabel("Window RMSE")
    ax.set_ylabel("Count")
    ax.set_title("signal_name")
    ax.legend()

    fig.tight_layout()
   
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return mean_rmse, q95_rmse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot test windows whose VAE reconstruction RMSE is near a target."
    )
    parser.add_argument("--config_file_path", required=True)
    parser.add_argument("--config_task_file_path", required=True)
 
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for plots (defaults to a target/tolerance-specific subdirectory).",
    )
    args = parser.parse_args()
    
    # Statistics
    RMSE = windows_rmse(args.config_file_path, args.config_task_file_path)
    rmse_stats =  rmse_statistics(RMSE)
    
    # Retrieve specific woindow
    window_index = rmse_stats["closest_to_mean"]["window_index"]
    shot_id = rmse_stats["closest_to_mean"]["shot_id"]
    reco, target = reconstruct_window(args.config_file_path, args.config_task_file_path, window_index, shot_id)
    
    # Visualize
    hist_rmse(RMSE, os.path.join(args.output_dir, f"RMSE_{RMSE['signal_name'][0]}.pdf"))

    if "psi" in RMSE['signal_name'][0]:
        plot_psi(
            reco,
            target,
            save_path=os.path.join(args.output_dir,f"{RMSE['signal_name'][0]}_reco_vs_target.pdf"),
            n_contours=15
            )

    with open(os.path.join(args.output_dir, "loss_curves.json"), 'r') as file:
        data = json.load(file)
    
    plot_loss_vs_epoch(data,title = RMSE['signal_name'][0], save_path = os.path.join(args.output_dir,f"{RMSE['signal_name'][0]}_losses_vs_epochs.pdf"))
if __name__ == "__main__":
    main()



