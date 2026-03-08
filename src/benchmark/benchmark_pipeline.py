import argparse
import json
import os
import sys
from torch.utils.data import DataLoader
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
                                          
from src.vae_pipeline.utils.utils import (
    read_data_split_csv, ComposeTransforms
)

from src.vae_pipeline.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)

from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.vae_pipeline.utils.utils import get_train_test_val_shots
from src.vae_pipeline.vae_pipeline import initialize_datasets, initialize_dataloaders
from src.vae_pipeline.collate_functions.collate_functions import WindowsCollate

from src.benchmark.utils import load_task_config, load_benchmark_settings, parse_args, load_vae_model
from src.vae_pipeline.models.vae_model import beta_VAE

def train_model(
    train_dataloader:DataLoader,
    vae_input_models: list[beta_VAE],
    vae_actuator_models: list[beta_VAE],
    vae_output_models: list[beta_VAE]
    ):
    
    for batch_idx, batch_ in enumerate(train_dataloader):
        breakpoint()
        
        print(f"\nBatch {batch_idx}")
        # print(batch_)
        shot_id, window_index, x_train, y_train = batch_

        print(f"The list of shot ID is  {batch_['shot_id'].item()}")
        print(f"The list of window Index is {batch_['window_index'].item()}")

        print(f"The x_train has been collated to shape (B, ..., T), , {[arr.shape for arr in batch_['x']]}")
        # print("Mean x_train", [torch.nanmean(arr) for arr in x_train])
        # print("Std x_train", [np.nanstd(arr) for arr in x_train])

        print(f"The y_train has been collated to shape (B, ..., T), {[arr.shape for arr in batch_['y']]}")
        
def main():
    args = parse_args()

    config_task_file_path: str = args.config_task_file_path
    config_model_file_path: str = args.config_model_file_path
    config_model_file_name: str = os.path.basename(config_model_file_path)

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
    
    # MAST datasets
    ###########################################################
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
    
    model_specific_transform = ModelSpecificTransform()
    train_model_dataset = initialize_TokaMark_dataset(
        dataset=base_train_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows = False,
        verbose=False
    )
    
    train_dataloader = DataLoader(
        dataset = train_model_dataset,
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
    if SETTINGS.LOCAL_PATHS.actuator_vae_models:
        for i, (source, signal_name) in enumerate(config_task["sources_and_signals"].get("actuator_name")):
            if signal_name not in SETTINGS.LOCAL_PATHS.actuator_vae_models[i]:
                raise ValueError("Rectify order of actuator_vae_models in confij.json to agree with the order in your task settings")
            
        vae_actuator_models = [load_vae_model(os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, this_vae)) 
                               for this_vae in SETTINGS.LOCAL_PATHS.actuator_vae_models]
    
    
    # Load output VAEs 
    # ----------------------------------------
    if SETTINGS.LOCAL_PATHS.output_vae_models:
        for i, (source, signal_name) in enumerate(config_task["sources_and_signals"].get("output_name")):
            if signal_name not in SETTINGS.LOCAL_PATHS.output_vae_models[i]:
                raise ValueError("Rectify order of output_vae_models in confij.json to agree with the order in your task settings")
        
        vae_output_models = [load_vae_model(SETTINGS.LOCAL_PATHS.vae_directory, this_vae) 
                            for this_vae in SETTINGS.LOCAL_PATHS.output_vae_models] 
    
    
    # Start training
    
    
if __name__ == "__main__":
    main()

    
    
    
    
    
    # train_MAST_dataset = initialize_MAST_dataset(
    #     config_task=config_task,
    #     shots_list=train_shots,
    #     local_flag=SETTINGS.local,
    #     use_std_scaling=True,
    #     return_incomplete_shots=True,
    #     remove_outliers=True,
    #     verbose=True
    # )
    # val_MAST_dataset = initialize_MAST_dataset(
    #     config_task=config_task,
    #     shots_list=val_shots,
    #     local_flag=SETTINGS.local,
    #     use_std_scaling=True,
    #     return_incomplete_shots=True,
    #     remove_outliers=True,
    #     verbose=True
    # )