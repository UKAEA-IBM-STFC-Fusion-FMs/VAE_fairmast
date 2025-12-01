import os
import sys
import yaml
import argparse

import pickle
import torch
import torch.multiprocessing as mp
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import (
    get_train_test_val_shots,
    initialize_datasets,
    initialize_dataloaders,
    ComposeTransforms,
)
from scripts.pipelines.utils.cnn_utils import (
    flatten_blocks,
    flatten_then_collate,
    get_cnn_order_scaling,
    inverse_standardize,
    plot_shot_gif,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import (
    DropSampleWithNans,
)
from scripts.pipelines.transforms.shot_level_transforms.cnn_transform import (
    CNNTransform,
)

from cnn_pipeline import create_cnn_architecture


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    # device = torch.device("mps")
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ======================================================================================================================
if __name__ == "__main__":
    print(f"\nNumber of available cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # GENERAL SETTINGS
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="/scripts/pipelines/configs/config_cnn_test.yaml",
        help="Path to the config YAML file",
    )
    args, unknown = parser.parse_known_args()

    print(args.config)
    # Load parameters from YAML configuration
    with open(REPO_ROOT + args.config, "r") as f:
        parameters = yaml.safe_load(f)
    print(parameters)  # optional, to verify contents

    # ..................................................................................................................
    # Parameters Setting

    # General parameters
    LOCAL_FLAG = parameters["local"]
    SUBSET_OF_SHOTS = parameters["subset_of_shots"]
    OUTPUT_FOLDER = REPO_ROOT + parameters["paths"]["data_output_directory"]
    RUN_EVALUATION = parameters["run_evaluation"]

    # Data
    source_signal_list = (
        parameters["input"]["data_names"] + parameters["input"]["target_names"]
    )

    # Preprocessing parameters
    ref_freq = parameters["ref_freq"]
    parameters_window_segmenter = parameters["window_segmenter_setting"]
    parameters_window_segmenter["x_keys"] = [
        f"{source}-{signal}" for source, signal in parameters_window_segmenter["x_keys"]
    ]
    parameters_window_segmenter["y_keys"] = [
        f"{source}-{signal}" for source, signal in parameters_window_segmenter["y_keys"]
    ]
    # Model Architecture parameters
    cnn_args = parameters["cnn_model"]

    # Dataloader paraneters
    dataloader_setting = parameters["dataloader_setting"]

    # ------------------------------------------------------------------------------------------------------------------
    # PRELIMINARY TASKS
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Preprocessing pipeline

    # Create sets of shot IDs for training, validation and testing
    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=SUBSET_OF_SHOTS
    )

    # Fit mean and std for signal transformation
    with open(
        REPO_ROOT + parameters["standardscaling_setting"]["mean_path"], "rb"
    ) as f:
        dict_mean = pickle.load(f)
    with open(REPO_ROOT + parameters["standardscaling_setting"]["std_path"], "rb") as f:
        dict_std = pickle.load(f)

    # Get the user-defined composite signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [
                StdScalingTransform(dict_mean[var], dict_std[var]),
                SamplingToReferenceTimeTransform(ref_freq),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    for var in [
        "magnetics-flux_loop_flux",
        "magnetics-b_field_pol_probe_ccbv_field",
        "magnetics-b_field_pol_probe_obr_field",
        "magnetics-b_field_pol_probe_obv_field",
        "magnetics-b_field_tor_probe_saddle_voltage",
    ]:
        signal_transform_map[var] = ComposeTransforms(
            [
                FillProfileWithZerosTransform(),
                StdScalingTransform(
                    dict_mean[var], dict_std[var]
                ),  # nan on one channel in std
                # FillProfileWithZerosTransform(),
                SamplingToReferenceTimeTransform(ref_freq),
            ]
        )

    for var in ["equilibrium-lcfs_r", "equilibrium-lcfs_z"]:
        signal_transform_map[var] = ComposeTransforms(
            [
                ReshapeLcfsTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                SamplingToReferenceTimeTransform(ref_freq),
            ]
        )

    # ..................................................................................................................
    # For CNN pipeline

    shot_transform = ComposeTransforms(
        [  # shape-consistent transform
            TruncationTransform(),
            WindowSegmenterTransform(
                **parameters_window_segmenter
            ),  # shape-modifying transform
            DropSampleWithNans(),
            CNNTransform(),  # shape-modifying transform
        ]
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots_, "val": val_shots_, "test": test_shots_},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True,
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=flatten_then_collate,
        **dataloader_setting,
        verbose=True,
    )
    test_dataloader = dataloaders_train_val_test["test"]

    # Create CNN architecture
    cnn_model = create_cnn_architecture(
        train_dataloader_=test_dataloader, **cnn_args, verbose=True
    )

    # Get data order for unstandardscaling
    shot_transform_without_CNN = ComposeTransforms(
        [  # shape-consistent transform
            TruncationTransform(),
            WindowSegmenterTransform(
                **parameters_window_segmenter
            ),  # shape-modifying transform
            DropSampleWithNans(),
            # CNNTransform()
        ]
    )
    order_var_for_inv_std = get_cnn_order_scaling(
        LOCAL_FLAG,
        train_shots_,
        source_signal_list,
        signal_transform_map,
        shot_transform_without_CNN,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CNN Visualization
    # ------------------------------------------------------------------------------------------------------------------

    # ..................................................................................................................
    # Load trained model

    # Load saved state
    best_model_path = OUTPUT_FOLDER + "best_model.pt"
    cnn_model.load_state_dict(
        torch.load(best_model_path, map_location=torch.device(device))
    )
    # Optional: move to device
    cnn_model.to(device)

    # ..................................................................................................................
    # Fixed visualization

    os.makedirs("shot_images", exist_ok=True)

    cnn_shot_dataset = MastDataset(
        local=LOCAL_FLAG,
        shots_list=test_shots_[0:50],
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=shot_transform,
    )
    shot_dataloader = DataLoader(
        cnn_shot_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=flatten_then_collate,
    )

    cnn_model.eval()  # Set model to eval mode
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    y_preds = []
    y_trues = []

    with torch.no_grad():
        
        for i, (batch) in enumerate(shot_dataloader):

            if batch is None:
                print(f"Cannot evaluate shot {i}")

            else : 
                x_test, y_test = batch

                # Convert inputs to tensors with batch dim = 1
                x_test = [arr.to(torch.float32).to(device) for arr in x_test]
                y_test = [arr.to(torch.float32).to(device) for arr in y_test]
                outputs = cnn_model(*x_test)

                pred = [arr.cpu().numpy() for arr in outputs]
                true = [arr.cpu().numpy() for arr in y_test]

                # Flatten each block into individual series
                flat_pred = flatten_blocks(pred)
                flat_true = flatten_blocks(true)

                new_flat_pred = inverse_standardize(
                    flat_pred, order_var_for_inv_std, dict_mean, dict_std
                )
                new_flat_true = inverse_standardize(
                    flat_true, order_var_for_inv_std, dict_mean, dict_std
                )

                avg_test_loss = [
                    torch.nn.MSELoss(reduction="mean")(pred, true).mean(dim=0).item()
                    for pred, true in zip(
                        [torch.from_numpy(arr).float().to(device) for arr in new_flat_pred],
                        [torch.from_numpy(arr).float().to(device) for arr in new_flat_true],
                    )
                ]

                # Save the gif
                plot_shot_gif(
                    new_flat_pred,
                    new_flat_true,
                    order_var_for_inv_std,
                    avg_test_loss,
                    i,
                    ref_freq,
                    out_dir=OUTPUT_FOLDER,
                )
        
