"""This script explores the latent representation learned by a Variational Autoencoder (VAE) 
    trained on MAST tokamak diagnostic signals. 
    
    The objective is to investigate whether the latent space can serve as a tool for identifying anomalous behaviour 
    and potential diagnostic inconsistencies.
    
    For a given shot, the code computes and visualises four complementary quantities as a function of time:
    
    1. Latent coordinates z(t) i.e., the evolution of each latent dimension is plotted versus time 
    to reveal how the VAE encodes changes in the plasma states.
    
    2. Latent-space velocity i.e., the distance travelled in latent space between consecutive time windows is computed. 
    Sudden increases may indicate rapid plasma transitions, abnormal operating regimes, or diagnostic faults.
    
    3. Reconstruction likelihood / anomaly score.  Quantities derived from the VAE objective (e.g. reconstruction error or ELBO) 
    are monitored over time to identify observations that deviate from the distribution learned during training.
    
    4. Probe-wise reconstruction residuals. 
    For each diagnostic channel, the difference between the measured signal and the VAE reconstruction is plotted. 
    Persistent or abrupt residuals may indicate sensor degradation, calibration drifts, saturation effects, or other inconsistencies 
    between a probe and the rest of the diagnostic set. Together, these visualisations provide
    a framework for analysing plasma trajectories in the learned latent space, 
    identifying anomalous events, and assessing the consistency of individual diagnostics with respect to 
    the collective behaviour observed by the VAE.
"""

import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import torch
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

from tokamark.tasks import get_task_metadata
from tokamark.data import initialize_TokaMark_dataset
from tokamark.tools.transforms.reshape_lcfs_transform import  ReshapeLcfsTransform

from src.utils.utils import (ComposeTransforms, get_train_test_val_shots, initialize_datasets, load_task_config)
from src.benchmark.utils import (process_data, masked_loss)

from src.common_transforms.general_transforms import ProbeDiagnosticTransform, StdScalingTransform

from src.probe_diagnostics.configs.config_setup import get_settings
from src.probe_diagnostics.utils import get_all_shots_for, find_outlayers, plot_latent_space_vs_shot_id, plot_residual_vs_shot_id
from src.vae_pipeline.models.vae_model import beta_VAE



def main():
    
    #### DETERMINE DEVICE ####################################
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--------------- RUNNING ON GPUs ---------------")
    else:
        device = torch.device("cpu")
        print(f"--------------- RUNNING ON CPUs ---------------")
    
    
    #### PARSE INPUT ####################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    parser.add_argument(
        "--config_task_file_path",
        default="",
        type=str,
        help="Path to configuration YAML task file."
    )
    
    args = parser.parse_args()
    
    
    #### LOAD CONFIG ####################################
    config_file_path = args.config_file_path
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
        try:
            SETTINGS = get_settings(config_file_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
            return 
    probe_name = SETTINGS.DATA.data_names[0]
    
    config_task_file_path: str = args.config_task_file_path
    try:
        config_task = load_task_config(config_task_file_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    dict_task_metadata = get_task_metadata(config_task,verbose=False)
    
    
    # MAKE LISTS OF DATA ####################################
    _, test_shots, _ = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = SETTINGS.TRAINING.num_test_samples,
        csv_path = SETTINGS.LOCAL_PATHS.data_split_csv_path
    )
    
    # filtered_shots = list(get_shots_for('generic_max_energy_time', "SW Beam"))
    import pandas as pd
    summary  = pd.read_parquet('https://mastapp.site/parquet/level2/shots')
    filtered =  list(summary.loc[ summary["generic_max_energy_time"] < 0.2, "shot_id"])

    common = [shot for shot in test_shots if shot in list(filtered)]
    if len(common)==0:
        raise ValueError(f"No shots available with the current filters")
    test_shots = common
    print(f"Nr shots = {len(test_shots)}")

    #### INITIALIZE DATA TRANSFORMS ####################################
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_signals_stats.yaml"), "r") as f:
        dict_stats_metadata = yaml.safe_load(f)

    # Signal-level transform map
    source_signal = f"{probe_name[0]}-{probe_name[1]}"
    if "lcfs" in probe_name:
        signal_transform_map = {
            source_signal : ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[source_signal]['mean'], dict_stats_metadata[source_signal]['std']),
                    ReshapeLcfsTransform()
                ]
            )
         
        }
    else:
        signal_transform_map = {
            source_signal : ComposeTransforms(
                [   
                    StdScalingTransform(dict_stats_metadata[source_signal]['mean'], dict_stats_metadata[source_signal]['std'])
                ]
            )
        }
      
    
    # INITIALIZE DATASET ####################################
    zarr_local_path = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2"
    if SETTINGS.DATA.local and zarr_local_path:
        store_mast_settings = {"base_local_zarr_path":zarr_local_path}  
    else :
        store_mast_settings = {"s3_mast_dataset_path": "mast/level2/shots/"}
        
    # store_mast_settings = {"s3_mast_dataset_path":"mast/level2/shots"} 
    base_datasets = initialize_datasets(
        sources_and_signals=SETTINGS.DATA.data_names,
        shots={"train": [], "val": [], "test": test_shots},
        signal_transform_map=signal_transform_map,
        local_flag=SETTINGS.DATA.local,
        store_mast_settings = store_mast_settings 
    )
    
    base_dataset = base_datasets['test']
    
    dataset = initialize_TokaMark_dataset(
        dataset=base_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=ProbeDiagnosticTransform(),
        test_mode=False,
        shuffle_windows = False,
        verbose=False
    )
    
    
    #### INITIALIZE DATALOADER ####################################
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = SETTINGS.TRAINING.dataloader_batch_size,
        num_workers =SETTINGS.TRAINING.num_workers,
        persistent_workers = False
    )
    
    #### LOAD BETA-VAE MODEL ####################################
    model = beta_VAE(SETTINGS)
    checkpoint = torch.load(SETTINGS.LOCAL_PATHS.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    model_type  = next(model.parameters()).dtype

    
    #### MAIN LOOP TO CREATE LATENT DATA DIC ####################################
    latent_data = defaultdict(dict)
    for batch_nr, batch in enumerate(dataloader):
        
        if batch_nr % 10 == 0:
            print(f"Batch nr = {batch_nr}")
        
        x = batch["x"].to(device=device, dtype=model_type)

        x_recon, mu, _, mask, x0 = model(x)
        
        mu = mu.detach().cpu()
        
        sum_mask = mask.sum()
        if sum_mask > 0:
            residual = (mask * abs(x_recon - x0)).sum() / sum_mask
            residual
        else:
            continue
        
        for shot_id, window_idx, latent in zip(
            batch["shot_id"],
            batch["window_index"],
            mu
        ):
            latent_data[int(shot_id)][int(window_idx)] = {
                "mu":latent.numpy(),
                "shot_average_residual": residual.item()
                }
  
  
    #### MAKE OUTPUT FOLDER
    output_directory = SETTINGS.LOCAL_PATHS.data_output_directory + probe_name[1] + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print( f"output_directory = {output_directory}")
   
   
    #### PLOTs
    shot_ids, mu_matrix, idx_0, idx_1, z_score  = plot_latent_space_vs_shot_id(latent_data, time_window_idx = 10, save_path = output_directory)
    plot_residual_vs_shot_id(latent_data, save_path = output_directory)
    

    #### SAVE FILES
    with open(os.path.join(output_directory,"outliers_latent_space.csv"), "w") as f:
        f.write("shot_id, latent_dim, mu, z_score \n")
        for s_idx, d_idx in zip(idx_0, idx_1):
            f.write(
                f"{shot_ids[s_idx]},"
                f"{d_idx},"
                f"{mu_matrix[s_idx, d_idx]:.6f},"
                f"{z_score[s_idx, d_idx]:.6f}\n"
            )
    with open(os.path.join(output_directory,"latent_representations.pkl"), "wb") as f:
        pickle.dump(dict(latent_data), f)
    
    

if __name__ == "__main__":
    main()
    
