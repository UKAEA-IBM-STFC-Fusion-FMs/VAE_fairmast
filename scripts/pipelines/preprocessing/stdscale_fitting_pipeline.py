import os
import sys

import pickle
import torch

import torch
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", "..", ".."
))  
print(REPO_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import read_data_split_csv

from scripts.pipelines.utils.utils import ComposeTransforms

from scripts.pipelines.transforms.signal_level_transforms.reshape_lcfs_transform import (
    ReshapeLcfsTransform
)


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


LOCAL_FLAG = True


# ..................................................................................................................
# For common pipeline

source_signal_list_ = [
    ('magnetics', 'flux_loop_flux'),
    ('magnetics', 'b_field_pol_probe_ccbv_field'),
    ('magnetics', 'b_field_pol_probe_obr_field'),
    ('magnetics', 'b_field_pol_probe_obv_field'),
    ('magnetics', 'b_field_tor_probe_saddle_voltage'),
    ('pf_active', 'coil_voltage'),
    ('pf_active', 'coil_current'),
    ('pf_active', 'solenoid_current'),
    ('pulse_schedule', 'i_plasma'),
    ('summary', 'power_nbi'),
    ('equilibrium', 'psi'),
    ('equilibrium', 'elongation'),
    ('equilibrium', 'elongation_axis'),
    ('equilibrium', 'triangularity_upper'),
    ('equilibrium', 'triangularity_lower'),
    ('equilibrium', 'lcfs_r'),
    ('equilibrium', 'lcfs_z'),
    ('equilibrium', 'x_point_r'),
    ('equilibrium', 'x_point_z'),
    ('equilibrium', 'minor_radius'),
    ('equilibrium', 'magnetic_axis_r'),
    ('equilibrium', 'magnetic_axis_z'),
    ('equilibrium', 'q95'),
    ('equilibrium', 'beta_tor'),
    ('equilibrium', 'beta_pol'),
    ('equilibrium', 'beta_normal'),
    ('equilibrium', 'bvac_rmag'),
    ('equilibrium', 'bphi_rmag'),
    ('summary', 'ip'),
    ('spectrometer_visible', 'filter_spectrometer_dalpha_voltage'), 
    ('soft_x_rays', 'horizontal_cam_lower'), 
    ('soft_x_rays', 'horizontal_cam_upper'), 
    ('pulse_schedule', 'n_e_line'), 
    ('magnetics', 'b_field_tor_probe_cc_field'), 
    ('magnetics', 'b_field_pol_probe_omv_voltage'), 
    ('thomson_scattering', 't_e'), 
    ('thomson_scattering', 'n_e')
]

train_sh, test_sh, val_sh = read_data_split_csv()


import numpy as np 

def collate_preprocessing (batch):
    
    formatted_batch = {f"{source}-{signal}": [] for source, signal in source_signal_list_}
    
    for data in batch :
        for var, data_var in data.items():
            if data_var['values'] is not None:
                formatted_batch[var].append(data_var['values'])

    formatted_batch_mean_std = {f"{source}-{signal}": [] for source, signal in source_signal_list_}

    for source, signal in source_signal_list_:
        var = f"{source}-{signal}"
        shapes = np.array([ar.shape for ar in [np.nanmean(arr, axis=-1) for arr in formatted_batch[var]]])
        uniq = np.unique(shapes, axis=0)
        if len(uniq)!=1:
        # if True:
            print(var)
            print(uniq)
            formatted_batch_mean_std[var] = [0, 0, 0]
        else:
            formatted_batch_mean_std[var] = [ len(formatted_batch[var]), 
                                            np.nanmean( np.stack( [np.nanmean(arr, axis=-1) for arr in formatted_batch[var]] ), axis = 0),
                                            np.nanmean( np.stack( [np.nanstd(arr, axis=-1) for arr in formatted_batch[var]] ), axis = 0) ]
    
    return(formatted_batch_mean_std)
    


import torch
from torch.utils.data import DataLoader

def compute_mean_std(dataset, batch_size=64, num_workers=4):

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        collate_fn=collate_preprocessing)

    n_samples = { f"{source}-{signal}" : 0 for source, signal in source_signal_list_ }
    sum_mean = { f"{source}-{signal}" : 0 for source, signal in source_signal_list_ }
    sum_std = { f"{source}-{signal}" : 0 for source, signal in source_signal_list_ }

    for batch in loader:

        for var in batch.keys():

            n_batch = batch[var][0]
            mean_batch = batch[var][1]
            std_batch = batch[var][2]

            sum_mean[var] += mean_batch * n_batch
            sum_std[var] += std_batch * n_batch

            n_samples[var] += n_batch

    mean = { var : 0.0 for var in batch.keys() }
    std = { var : 0.0 for var in batch.keys() }

    for var in batch.keys():
        # mean
        mean[var] = sum_mean[var] / n_samples[var]
        where_are_NaNs = np.isnan(mean[var])
        mean[var][where_are_NaNs] = 0
        # std
        std[var] = sum_std[var] / n_samples[var]
        where_are_NaNs = np.isnan(std[var])
        std[var][where_are_NaNs] = 1


    return mean, std


# ======================================================================================================================
if __name__ == "__main__":

    # ..................................................................................................................
    # Specific Data Preprocessing for LCFS profiles 

    signal_transform_map = {
        var: ComposeTransforms([]) 
        for var in [f'{source}-{signal}' for source, signal in source_signal_list_] }

    signal_transform_map['equilibrium-lcfs_r'] = ComposeTransforms([ReshapeLcfsTransform()])
    signal_transform_map['equilibrium-lcfs_z'] = ComposeTransforms([ReshapeLcfsTransform()])

    preprocessing_train_dataset = MastDataset(
        local=LOCAL_FLAG,
        shots_list=train_sh,
        # shots_list=train_sh[0:10],
        source_signal_list=source_signal_list_,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=None
    )

    BATCH_SIZE = 100
    NUM_WORKERS = 8
    # BATCH_SIZE = 3
    # NUM_WORKERS = 0

    mean, std = compute_mean_std(preprocessing_train_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=NUM_WORKERS)
    
    with open('preprocessing/dict_mean_shot.pkl', 'wb') as f_:
        pickle.dump(mean, f_)
    with open('preprocessing/dict_std_shot.pkl', 'wb') as f_:
        pickle.dump(std, f_)
    
