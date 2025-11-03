import argparse
import json
import os
import sys
import pickle
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict
import psutil
import time
import subprocess
import random

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from scripts.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms, load_models, to_dict
)
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import (
    get_mean_shot,
    get_std_shot,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_with_zeros_imputer_transform import (
    FillWithZerosImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.forward_fill_imputer_transform import (
    ForwardFillImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform

from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.models.conv1d_vae_model import Conv1dVAE
from scripts.pipelines.models.conv1d_vae_model import loss_function
from scripts.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.collate_functions.collate_functions import Conv1dVAECollate

# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

def get_train_test_val_shots(max_index=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        train_sh = train_sh[:max_index]
        val_sh = val_sh[:max_index]
        test_sh = test_sh[:max_index]

    return train_sh, test_sh, val_sh

def fit_mean_and_std_for_signal_transform( 
                                          train_shots,
                                          output_dir, 
                                          verbose=False,
                                          use_existing=False, 
                                          local=True
                                          ):
    """
    Fit or load mean and std for signal transformation.

    Args:
        output_sub_dir: Directory to save/load fitted parameters
        verbose: Print verbose output
        use_existing: If True, try to load existing fitted parameters instead of re-fitting
    """
    os.makedirs(output_dir, exist_ok=True)

    mean_path = os.path.join(output_dir, "dict_mean_shot.pkl")
    std_path = os.path.join(output_dir, "dict_std_shot.pkl")

    # Try to load existing files if requested
    if use_existing and os.path.exists(mean_path) and os.path.exists(std_path):
        if verbose:
            print("\n\n----------LOADING EXISTING FITTED PARAMETERS----------\n")
            print(f"Loading fitted parameters from: {output_dir}")

        try:
            with open(mean_path, "rb") as f:
                dict_mean_ = pickle.load(f)
            with open(std_path, "rb") as f:
                dict_std_ = pickle.load(f)

            if verbose:
                print(f"Successfully loaded mean and std dictionaries")
                print(f"Mean dict keys: {list(dict_mean_.keys())}")
                print(f"Std dict keys: {list(dict_std_.keys())}")

            return dict_mean_, dict_std_

        except Exception as e:
            if verbose:
                print(f"Error loading existing fitted parameters: {e}")
                print("Falling back to re-fitting")

    if verbose:
        print("\n\n----------TRANSFORM FITTING----------\n")

    preprocessing_train_dataset = MastDataset(
        local=local,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None,
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    dict_std_ = get_std_shot(preprocessing_train_dataset)

    # Save fitted parameters
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir}")

    with open(mean_path, "wb") as f:
        pickle.dump(dict_mean_, f)
    with open(std_path, "wb") as f:
        pickle.dump(dict_std_, f)

    return dict_mean_, dict_std_

def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False
    ):
    
    datasets_ = {"train": None, "val": None, "test": None}
    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if shots[data_set_type]:
            datasets_[data_set_type] = MastDataset(
                local=local_flag,
                shots_list=shots[data_set_type],
                source_signal_list=sources_and_signals,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transforms,
            )
    # datasets_["train"] = CachedDataset(datasets_["train"])
    # datasets_["val"]   = CachedDataset(datasets_["val"])
    # datasets_["test"]  = CachedDataset(datasets_["test"])
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers=True
    ):
    
    dataloaders_ = {"train": None, "val": None, "test": None}

    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if datasets[data_set_type]:
            dataloaders_[data_set_type] = DataLoader(
                dataset=datasets[data_set_type],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_function,
            )

    return dataloaders_

def time_access(dataset, indices):
    start = time.time()
    for i in range(len(indices)):
        if i%10 ==0:
            print(i)
        _ = dataset[i]
    return time.time() - start

def test_dataset_speed_access(dataset, indices):
    t_seq = time_access(dataset,indices)
    random.shuffle(indices)
    t_rand = time_access(dataset,indices)

    print(f"Sequential read time: {t_seq:.2f}s")
    print(f"Random read time: {t_rand:.2f}s")


def test_dataloader(
    SETTINGS,
    train_dataloader, 
    device,
    verbose=True):
    
    times_vs_epoch = []
    for epoch in range(SETTINGS.TRAINING.num_epochs):
        verbose and print(f"\nEpoch {epoch+1}\n")
        
        average = 0.0
        counter = 0
        times = []
        start = time.time()
        for batch_idx, batch in enumerate(train_dataloader):
            elapsed = time.time()- start
            print(f"Batch idx: {batch_idx}, DatLoader Elapsed Time {elapsed}")
            times.append(elapsed)
            # time.sleep(5)
            counter +=1
            average += elapsed
            start = time.time()
        print(f"Average time epoch {epoch}: {elapsed/counter}")
        times_vs_epoch.append(times)
    return times_vs_epoch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "scripts/pipelines/configs/config.json",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    config_file_path = args.config_file_path
    cofig_file_name = os.path.basename(config_file_path)
    
    # Load configuration from JSON file
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
         SETTINGS = get_settings(config_file_path)  
    
    
    # HPC settings for CPUs only
    num_workers = SETTINGS.TRAINING.num_workers
    mp.set_start_method("spawn", force=True)

    # Output data folder
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_" + cofig_file_name.removesuffix(".json")+ "/"
    print( f"output_directory = {output_directory}")
    source_signal_list = SETTINGS.DATA.data_names

    # Parameters for window segmentation (no x/y split for VAE)
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,  # 100ms windows
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "min_samples_per_window": SETTINGS.TIME_SEGMENTATION.min_samples_per_window,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        SETTINGS.TRAINING.num_train_samples
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        train_shots,
        output_directory,
        verbose=False,
        use_existing=SETTINGS.BETA_VAE.existing_fitted_params,
        local = SETTINGS.DATA.local
    )

    # model_dictionary = load_models(SETTINGS.DATA.data_names, 
    #                                SETTINGS.LOCAL_PATHS.joblib_directory)
    
    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                ForwardFillImputerTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                #ImputerTransform(model_dictionary["imputer"][var], 
                         #SETTINGS.LOCAL_PATHS.average_values_file_path),
                FillWithZerosImputerTransform(),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transforms = ComposeTransforms(
        [
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            Conv1dVAETransform()
        ]
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local
    )
    
    shuffle = True
    signals_to_collate = [f"{source}-{signal}" for source, signal in source_signal_list]
    conv1d_vae_collate_fn = Conv1dVAECollate(signals_to_collate, SETTINGS.TRAINING.dataloader_batch_size)
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]

    if 0:
        start = time.time()
        iterator = iter(train_dataloader)
        print(f"Iterator creation time: {time.time() - start:.4f} seconds")
        for i in range(20):
            try:
                start = time.time()
                batch = next(iterator)
                print(f"Batch {i} load time: {time.time() - start:.4f} seconds")
            except Exception:
                break

    if 1:
        times_vs_epoch = test_dataloader(SETTINGS, train_dataloader, device)
        
        fig, ax = plt.subplots()
        
        for times in times_vs_epoch:
            x = np.arange(len(times))
            y = np.array(times)
            ax.plot(x, y, linestyle='-',color='blue', marker='o')
        
        ax.set_ylabel('Sec')
        ax.set_xlabel('Batch')
        ax.set_title(f'Batch size {SETTINGS.TRAINING.dataloader_batch_size}')
        plt.show()
        fig.savefig(f'DataLoaderTime_{SETTINGS.TRAINING.num_train_samples}_shuffl_{shuffle}.pdf')
            
    if 0:
        test_dataset_speed_access(datasets_train_val_test['train'],train_shots)  