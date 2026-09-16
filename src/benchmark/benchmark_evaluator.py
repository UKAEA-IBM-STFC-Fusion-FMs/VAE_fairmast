
'''
python src/benchmark/benchmark_evaluator.py --config_benchmark_file_path src/benchmark/configs/task1_3_config_gamma_factor.json --config_task_file_path tokamark/src/tokamark/tasks_configs/group_1_reconstruction/task_1-3.yaml
'''

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
from tokamark.evaluator import (WindowMetricsAccumulator, compute_metrics)

from src.benchmark.benchmark_model import BenchmarkModel                    
from src.benchmark.utils import (
                                load_benchmark_settings, 
                                parse_args,
                                create_vae_dictionary,
                                process_data, 
                                masked_loss
                                )
from src.benchmark.configs.benchmark_setup import SettingsBenchmark
from src.benchmark.benchmark_visualization import align_shapes

from src.utils.utils import ( load_task_config, ComposeTransforms, get_train_test_val_shots, initialize_datasets)
from src.utils.layer_factory import SequentialBuilder
from src.common_transforms.general_transforms import ModelSpecificTransform, StdScalingTransform



def benchmark_eval():
    # --------------------------------------------------------------------------------------
    # Find device 
    # --------------------------------------------------------------------------------------   
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--------------- RUNNING ON GPUs ---------------")
    else:
        device = torch.device("cpu")
        print(f"--------------- RUNNING ON CPUs ---------------")

        

    # --------------------------------------------------------------------------------------
    # Parse arguments and load config files
    # --------------------------------------------------------------------------------------   
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


    # --------------------------------------------------------------------------------------
    # Prepare Output
    # --------------------------------------------------------------------------------------   
    # Check output directory
    output_directory = SETTINGS.LOCAL_PATHS.output_directory + config_benchmark_file_name.removesuffix(".json") + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
    

    # Initialize task specific metadata
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )
    

    # --------------------------------------------------------------------------------------
    # Prepare Inputs
    # --------------------------------------------------------------------------------------   
    # Get source-signal
    source_signal_list = (
        (config_task["sources_and_signals"].get("input_name") or [])
        + (config_task["sources_and_signals"].get("actuator_name") or [])
        + (config_task["sources_and_signals"].get("output_name") or [])
    )

    # Reinforce uniqueness
    source_signal_list = [
        s for i, s in enumerate(source_signal_list) if s not in source_signal_list[:i]
    ]    

    # Open file containing mean and std values
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_signals_stats.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)



    # --------------------------------------------------------------------------------------
    # Create transform map
    # --------------------------------------------------------------------------------------  
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
    
    # --------------------------------------------------------------------------------------
    #  Prepare DataSet and DataLoader
    # --------------------------------------------------------------------------------------   
    # Get lists of shot IDs test samples
    _, test_shots, _ = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = None,
        csv_path = SETTINGS.LOCAL_PATHS.data_split_csv_path)

    # Prepare base dataset
    zarr_local_path = "/lustre/home/bf3280/tokamark_fairmast_dataset"
    store_mast_settings = {"base_local_zarr_path":zarr_local_path} if SETTINGS.local and zarr_local_path else None
    base_datasets = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": [], "val": [], "test": test_shots},
        signal_transform_map=signal_transform_map,
        local_flag=SETTINGS.local,
        cache_data=False,
        store_mast_settings=store_mast_settings
    )

    base_test_dataset = base_datasets['test']

    # Specific dataset (TokaMark_dataset)
    model_specific_transform = ModelSpecificTransform()
    test_model_dataset = initialize_TokaMark_dataset(
        dataset=base_test_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows = False,
        verbose=False
    )
    
    # DataLoader
    test_dataloader = DataLoader(
        dataset = test_model_dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =  SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )


    # --------------------------------------------------------------------------------------
    # Load VAEs and move them to device
    # --------------------------------------------------------------------------------------  
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
            

    # --------------------------------------------------------------------------------------
    # Initialize benchmark model load checkpoint and move it to device
    # --------------------------------------------------------------------------------------
    model = BenchmarkModel(SETTINGS)
    model_path = os.path.join(output_directory, "best_model.pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
        
    # --------------------------------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------------------------------
    task_name = config_task["task_name"]
    output_feature_names = config_task["sources_and_signals"].get("output_name") or []

    accumulator = WindowMetricsAccumulator(task_name)
    benchmark_evaluation_per_shot(
        test_dataloader=test_dataloader,
        model=model,
        vae_dictionary=vae_dictionary,
        dict_stats_metadata=dict_stats_metadata,
        output_feature_names=output_feature_names,
        accumulator=accumulator,
        verbose=True
    )

    # Compute and save metrics
    results_dir = os.path.join(output_directory, "results_metrics")
    df_task_metrics = compute_metrics(
        task=task_name,
        output_dir=results_dir,
        window_metrics_accumulator=accumulator,
        save_windows_metrics=True,
        save_shot_metrics=True,
        save_task_metrics=True,
    )

    print(f"Task metrics for {task_name} saved under {os.path.join(results_dir, task_name)}/")
    print(df_task_metrics)



def split_and_decode_outputs(reconstruction, target_data_list, output_vaes):
    """Split the concatenated model output into per-signal chunks and decode
    each chunk back to real (still standardized) space.

    The model output columns follow the order of `target_data_list`, since the
    model is trained against `torch.cat(target_data_list, dim=1)`.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Model output, shape [B, sum_i width_i].
    target_data_list : list[torch.Tensor]
        Per-signal processed targets (encoded latents or flattened reals).
    output_vaes : list[beta_VAE | None]
        One output VAE per signal (None when the signal is kept in real space).

    Returns
    -------
    list[torch.Tensor]
        One real-space (standardized) tensor per output signal.
    """

    widths = [t.shape[1] for t in target_data_list]
    chunks = torch.split(reconstruction, widths, dim=1)

    signals_real_space = []
    for chunk, vae in zip(chunks, output_vaes):
        if vae is not None:
            p = next(vae.parameters())
            signals_real_space.append(vae.decode(chunk.to(device=p.device, dtype=p.dtype)))
        else:
            signals_real_space.append(chunk)

    return signals_real_space


# ----------------------------------------------------------------------------------------------------------------------

def benchmark_evaluation_per_shot(
    test_dataloader,
    model,
    vae_dictionary,
    dict_stats_metadata,
    output_feature_names,
    accumulator,
    verbose=True
):
    """Evaluate the benchmark model over the test set and accumulate per-window
    RMSE/MAE (in physical units) into the given WindowMetricsAccumulator.

    Parameters
    ----------
    test_dataloader : DataLoader
        Dataloader yielding batches with keys 'x', 'y', 'shot_id', 'window_index'.
    model : nn.Module
        Trained benchmark model (BenchmarkModel if use_mask, else single MLP).
    vae_dictionary : dict
        {"input": {...}, "actuator": {...}, "output": {...}} of loaded VAEs.
    dict_stats_metadata : dict
        Signal stats from dict_signals_stats.yaml (mean/std per 'source-signal').
    output_feature_names : list[list[str]]
        Output signals as [source, signal] pairs (task config 'output_name').
    accumulator : WindowMetricsAccumulator
        Accumulator for the task being evaluated.
    verbose : bool
        If True, print progress.
    """

    input_vae = list(vae_dictionary["input"].values()) + list(vae_dictionary["actuator"].values())
    output_vaes = list(vae_dictionary["output"].values())

    if len(vae_dictionary["input"]) == 0:
        raise ValueError("Input VAE is required.")

    model.eval()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch is None:
                continue

            if batch_idx % 100 == 0 and verbose:
                print(f"\nBatch {batch_idx}")

            # Batch content
            x = batch["x"]
            y = batch["y"]
            shot_id = batch["shot_id"]
            window_index = batch["window_index"]

            # Align dtype/device with the reference input VAE
            p = next(input_vae[0].parameters())

            input_ = [
                t if (t.device == p.device and t.dtype == p.dtype) else t.to(device=p.device, dtype=p.dtype)
                for t in x
            ]
            target_ = [
                t if (t.device == p.device and t.dtype == p.dtype) else t.to(device=p.device, dtype=p.dtype)
                for t in y
            ]

            # Standardized real-space targets, before process_data modifies them
            target_original = [t.clone() for t in target_]

            # Encode inputs / targets (VAE latents or flattened real space)
            input_data_list, input_mask_list = process_data(input_vae, input_, "do_not_expand_mask_over_latent_dim")
            target_data_list, target_mask_list = process_data(output_vaes, target_, "expand_mask_over_latent_dim")

            input_data = torch.cat(input_data_list, dim=1)
            input_mask = torch.cat(input_mask_list, dim=1)

            target_data = torch.cat(target_data_list, dim=1)  # [B, sum n_signals]
            target_mask = torch.cat(target_mask_list, dim=1)  # [B, sum n_signals]


            # Model prediction
            reconstruction = model(input_data, input_mask)

            if not bool(torch.isfinite(reconstruction).all()):
                if verbose:
                    print(f"[batch {batch_idx}] non-finite reconstruction; skipping sub-batch.")
                continue

            # Back to real space (still standardized)
            reco_real_space = split_and_decode_outputs(reconstruction, target_data_list, output_vaes)
            reco_real_space, target_real_space = align_shapes(reco_real_space, target_original)

            # Per-signal metrics in physical units
            for i, (source, signal) in enumerate(output_feature_names):
                key = f"{source}-{signal}"
                mean = dict_stats_metadata[key]["mean"]
                std = dict_stats_metadata[key]["std"]

                y_pred = reco_real_space[i].detach().cpu().numpy().astype(np.float64)
                y_target = target_real_space[i].detach().cpu().numpy().astype(np.float64)
                
                y_pred = y_pred * std + mean
                y_target = y_target * std + mean

                b = y_pred.shape[0]
                accumulator.add_batch(
                    y_target=y_target.reshape(b, -1),
                    y_pred=y_pred.reshape(b, -1),
                    shot_ids=shot_id,
                    window_indices=window_index,
                    feature_name=key,
                )

    print("Evaluation done. Per-window metrics accumulated.")


if __name__ == "__main__":
    benchmark_eval()