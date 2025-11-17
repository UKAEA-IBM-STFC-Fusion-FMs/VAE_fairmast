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
from scripts.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform

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
                                          source_signal_list,
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "scripts/pipelines/configs/config12.json",
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

    # Output data folder
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_" + cofig_file_name.removesuffix(".json")+ "/"
    source_signal_list = SETTINGS.DATA.data_names

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        SETTINGS.TRAINING.num_train_samples
    )

    # Fit mean and std for signal transformation
    dict_mean_, dict_std_ = fit_mean_and_std_for_signal_transform(
        train_shots,
        output_directory,
        source_signal_list,
        verbose=False,
        use_existing=SETTINGS.BETA_VAE.existing_fitted_params,
        local = SETTINGS.DATA.local
    )

    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
        dict_mean = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
        dict_std = pickle.load(f)

    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_mean[var], dict_std[var]),
                ImputerTransform()
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    signal_transform_map_2 = {
        var: ComposeTransforms(
            [   
                ForwardFillImputerTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillWithZerosImputerTransform(),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    datasets1  = MastDataset(
                local=SETTINGS.DATA.local,
                shots_list= val_shots,
                source_signal_list=source_signal_list,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=None,
            )
    datasets2  = MastDataset(
                local=SETTINGS.DATA.local,
                shots_list= val_shots,
                source_signal_list=source_signal_list,
                signal_level_transform_map=signal_transform_map_2,
                shot_level_transform=None,
            )
    
    for i in range(10):
        breakpoint()
        data1 =  datasets1[i]
        data2 =  datasets2[i]
