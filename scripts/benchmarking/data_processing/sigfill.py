"""
Classes for filling missing data in MAST signals.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
import sys


cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd)
scripts_dir = os.path.join(os.path.dirname(mother_dir))
sys.path.append(scripts_dir)

from MAST_tools.signal_utils import MASTSignalManager  
from MAST_tools.store_utils import MASTStorageManager
from utils import (is_finite_numeric_array,
                    read_signals,
                    make_dataframe_from_shot_ids,
                    shuffle_shot_ids)
from pipelines.utils.utils import read_data_split_csv
from config_files.config_setup import get_settings

class SimpleFiller:
    def __init__(self, vals):
        self.vals = vals

    def simple_fill(self):
        """_Process an array of values with missing data,
        and return the imputed array.
        
        Returns
        -------
        Imputed values as numpy array.
        """

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp.fit(self.vals)

        return imp.transform(self.vals)


class ConditionalFiller:

    def __init__(self, shot_tab_list, col_keys_list, col_means=None, col_covars=None, random=-1):
        self.shot_tab_list = shot_tab_list
        self.col_keys_list = col_keys_list
        self.col_means = col_means
        self.col_covars = col_covars
        if random < 0:
            self.deterministic = True
        else:
            self.deterministic = False
            np.random.seed(random)

    def subcov(self, col_covars, list_1, list_2):
        """Extract submatrix from full covariance matrix"""
        sub = np.array([[col_covars[i][j] for j in list_2] for i in list_1])
        return sub


    def stack_shots(self, shot_tab_list):
        return pd.concat(shot_tab_list, axis=0, ignore_index=True)

    def masked_mean(self, shot_tab_list, col_keys_list):
        all_data = self.stack_shots(shot_tab_list)
        return {col: all_data[col].mean(skipna=True) for col in col_keys_list}

    def masked_covar(self, shot_tab_list, col_keys_list):
        all_data = self.stack_shots(shot_tab_list)
        return all_data[col_keys_list].cov(min_periods=1).fillna(0).to_dict()

    def fit_mu_cov(self, shot_tab_list=None, col_keys_list=None, fit_mu=True, fit_cov=True):
        if shot_tab_list is None:
            shot_tab_list = self.shot_tab_list
        if col_keys_list is None:
            col_keys_list = self.col_keys_list
        else:
            self.col_keys_list = col_keys_list

        self.allshots_tab = self.stack_shots(shot_tab_list)

        col_means = self.masked_mean(shot_tab_list, col_keys_list)
        col_covars = self.masked_covar(shot_tab_list, col_keys_list)

        if fit_mu:
            self.col_means = col_means
        if fit_cov:
            self.col_covars = col_covars

    def fill_shots(self, 
    shot_tab_list=None, 
    col_keys_list=None, 
    col_means=None, 
    col_covars=None,
    missing_val=np.nan,
    focused_shot_index = 0):

        if shot_tab_list is None:
            shot_tab_list = self.shot_tab_list
        if col_keys_list is None:
            col_keys_list = self.col_keys_list
        if col_means is None:
            col_means = self.col_means
        if col_covars is None:
            col_covars = self.col_covars

        if focused_shot_index > len(shot_tab_list):
            print(f"Error: focused_shot_index > len(shot_tab_list)")
            return None

        filled_list = []

        for nr, tshot in enumerate(shot_tab_list):
            
            if nr != focused_shot_index:
                continue

            tshot_copy = tshot.copy()
            missing_cols = [col for col in col_keys_list if tshot[col].isna().any()]
            given_cols = [col for col in col_keys_list if not tshot[col].isna().any()]

            if len(missing_cols) ==0:
                filled_list.append(tshot_copy)
                continue

            # Build conditional distribution
            sigma_11 = self.subcov(col_covars, missing_cols, missing_cols)
            sigma_00 = self.subcov(col_covars, given_cols, given_cols)
            sigma_10 = self.subcov(col_covars, missing_cols, given_cols)

            X_given = tshot_copy[given_cols].to_numpy()
            mu_given = np.array([col_means[col] for col in given_cols])
            mu_missing = np.array([col_means[col] for col in missing_cols])

            dX_given = X_given - mu_given

            # Estimate conditional mean of missing variables
            X_missing_t = mu_missing.reshape(-1, 1) + sigma_10 @ np.linalg.pinv(sigma_00) @ dX_given.T

            if self.deterministic:
                X_missing_eps = np.zeros_like(X_missing_t)
            else:
                cov_post = sigma_11 - sigma_10 @ np.linalg.pinv(sigma_00) @ sigma_10.T
                X_missing_eps = np.random.multivariate_normal(np.zeros(len(missing_cols)), cov_post)

            X_missing = X_missing_t + X_missing_eps

            # Fill the missing values
            for idx, col in enumerate(missing_cols):
                tshot_copy[col] = X_missing[idx]

            filled_list.append(tshot_copy)

        return filled_list
    
        
    def conditional_fill(self, focused_shot_index: int):
        """Apply fill_shots method to a specific shot.

        Parameters
        ----------
        focused_shot_index : the positional index of the shot of which we wish 
        to impute missing components within the self.shot_tab_list

        Returns
        -------
        Filled shots
        """

        try:
            self.fit_mu_cov()
            filled_shots = self.fill_shots(focused_shot_index=focused_shot_index)
            return filled_shots

        except Exception as e:
            print(f"Exception {e}")
            return None
    
        
def run_simple_filler(
    shot_id,
    source,
    signal_name, 
    store_manager,
    local
    ):

    sig = MASTSignalManager()

    store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})

    sig_values = sig.get_signal_values(
        data_origin=store,
        source_name=source,
        signal_name=signal_name
    )

    if sig_values is not None:
        imputer = SimpleFiller(sig_values)
        return imputer.simple_fill()
    else:
        return None

def run_conditional_fill(local, 
                            source, 
                            signal_name, 
                            shot_ids,
                            focused_shot_index):
    store_manager = MASTStorageManager()

    channels, df = make_dataframe_from_shot_ids(store_manager,
                    shot_ids,
                    source, 
                    signal_name,
                    local=local)

    filler = ConditionalFiller(df, channels)
    filled_shots = filler.conditional_fill(focused_shot_index)
    
    return channels, filled_shots


def signals_average_across_shots_v0(store_manager : MASTStorageManager, 
                                 shot_ids : list[int],
                                 filepath:str,
                                 local : bool,
                                 json_filename: str = "averaged_arrays.json"):

    """
    For each shot:

    1- considers all signal vectors listed in the 
    filepath

    2- calculates the average value of each channel 
    in the signal. 

    3- Set to zero channels that contain NaN;

    3- returns the average value across shots in a json format:
    {
        "data": [
            {
                "signal_name": [ 
                    [avergae channel 0 ],
                    [average channel 1],
                    ...
                ],
                ...
            }
        ]
    }
    """

    # Read file containing signal names and the number of channels for each signal
    data_set = read_signals(filepath)

    data_set_counter = {}
    data_set_sums = {}

    # Loop through shots
    for nr,shot_id in enumerate(shot_ids):

        #if nr>=0 and nr%10 == 0:
        #    print(f"Shot number {nr+1} out of {len(shot_ids)}")

        store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})

        for key, value in data_set.items():
            # Retrieve signal source and name
            source_name, signal_name = key.split("-")

            # Form instance of SIGNAL
            sig = MASTSignalManager()

            # Prepare tracking
            data_set_counter.setdefault(signal_name, np.zeros(value))
            data_set_sums.setdefault(signal_name, np.zeros(value))

            try:
                vals = sig.get_signal_values(
                    data_origin=store,
                    source_name=source_name,
                    signal_name=signal_name
                )

                mean_vals = np.mean(vals, axis=1) # Returns NaN if NaN is in vals

                # Check mean_vals are all numeric different from NaN and Inf
                for i, mean_val in enumerate(mean_vals):

                    if is_finite_numeric_array(mean_val):
                        data_set_counter[signal_name][i] += 1
                    else:
                        mean_vals[i]=0

                # Add mean_vals to data_set_sums
                data_set_sums[signal_name] += mean_vals

            except Exception as e:
                print(f"Exception in cycle: {nr}, shot ID: {shot_id}")
                print(f"Exception: {e}")

    # Calculate average 
    data = []
    for signal_name, summed_array in data_set_sums.items():
        
        counter = data_set_counter[signal_name]
        
        averaged_channels = []
        for channel, count in zip(summed_array, counter):
            if count> 0:
                averaged_channels.append(channel / count)
            else:
                print(f"No valid data to average for value '{signal_name}'.")

        data.append({signal_name: averaged_channels})
     
    json_data = {"data":data}
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)


def signals_average_across_shots_v1(store_manager : MASTStorageManager, 
                                 shot_ids : list[int],
                                 data_set: list[list[str,float]],
                                 local : bool,
                                 json_filename: str = "averaged_arrays.json"):

    """
    For each shot:

    1- considers all signal vectors listed in the 
    filepath

    2- calculates the average value of 
    finite, non-NaN elements accross each channel 
    accross all shots. 

    3- Set to 
    - nan channels that don't contain finite values;
    - inf where overflow happens during the calculations.

    3- returns the average value across shots in a json format:
    {
        "data": [
            {
                "signal_name": [ 
                    [avergae channel 0 ],
                    [average channel 1],
                    ...
                ],
                ...
            }
        ]
    }
    """


    data_set_counter = {}
    data_set_sums = {}
    data_time_resolution = {}
    
    # Loop through shots
    for nr,shot_id in enumerate(shot_ids):
        store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})

        for key, value in data_set:
            # Retrieve signal source and name
            source_name, signal_name = key.split("-")

            # Form instance of SIGNAL
            sig = MASTSignalManager()

            # Prepare tracking
            data_set_counter.setdefault(signal_name, np.zeros(value))
            data_set_sums.setdefault(signal_name, np.zeros(value))
            try:
                vals = sig.get_signal_values(
                    data_origin=store,
                    source_name=source_name,
                    signal_name=signal_name
                )

                times, _ = sig.get_signal_times_and_time_type(
                    signal_name=signal_name,
                    data_origin= store,
                    source_name=source_name
                )

                data_time_resolution[signal_name] = abs(times[1]-times[0])

                data_is_finite = np.isfinite(vals)
                num_vals = np.sum(data_is_finite,axis=1)
                try:
                   sum_vals = np.sum(vals,axis=1,where=data_is_finite)
                except OverflowError:
                    print("Exception1 in signals_average_across_shots_v1")
                    print(f"Exception in cycle: {nr}, shot ID: {shot_id},  for {source_name}, {signal_name}")
                    print(f"Exception: {e}")
                    sum_vals = np.inf 

                # Add mean_vals to data_set_sums
                data_set_counter[signal_name] += num_vals
                data_set_sums[signal_name] += sum_vals
                    

            except Exception as e:
                print("Exception2 in signals_average_across_shots_v1")
                print(f"in cycle: {nr}, shot ID: {shot_id},  for {source_name}, {signal_name}")
                print(f"Exception: {e}")
                print(f"shape of vals is {np.shape(vals)}")

    # Calculate average 
    data = []
    for signal_name, summed_array in data_set_sums.items():
        
        counter = data_set_counter[signal_name]
        
        averaged_channels = []
        for channel, count in zip(summed_array, counter):
            if count> 0:
                averaged_channels.append(channel / count)
            else:
                averaged_channels.append(np.nan)
                print(f"No valid data to average for value '{signal_name}'.")

        data.append({
            signal_name: averaged_channels,
            "time_resolution": data_time_resolution[signal_name]
            })
     
    json_data = {"data":data}
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)

def shotids_not_fully_nan_signals(store_manager : MASTStorageManager, 
                                 shot_ids : list[int],
                                 data_set: list[list[str,float]],
                                 local : bool,
                                 json_filename: str = "signal_shotids_not_fully_nan.json"):

    """
    For each shot:

    1- considers all signal vectors listed in the 
    filepath

    2- check if all elements are finite, appends shot_id if yes

    3- returns the average value across shots in a json format:
    {
        "data": [
            {
                "signal_name": [ 
                        list_of_shot_ids 
                ],
                ...
            }
        ]
    }
    """

    data_set_idx = {}

    # Loop through shots
    for nr,shot_id in enumerate(shot_ids):

        store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})

        for key, value in data_set:
            # Retrieve signal source and name
            source_name, signal_name = key.split("-")
            data_set_idx.setdefault(signal_name, [])

            # Form instance of SIGNAL
            sig = MASTSignalManager()

            # Prepare tracking
            try:
                vals = sig.get_signal_values(
                    data_origin=store,
                    source_name=source_name,
                    signal_name=signal_name
                )

                
                if not np.any(np.all(~np.isfinite(vals), axis=1)): 
                    data_set_idx[signal_name].append(shot_id)
            except Exception as e:
                print("Exception1 in shotids_not_fully_nan_signals")
                print(f"Exception in cycle: {nr}, shot ID: {shot_id}, for {source_name}, {signal_name}")
                print(f"Exception: {e}")
                print(f"shape of vals is {np.shape(vals)}")

    json_data = {"data":data_set_idx}

    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)


def test_signals_average_across_shots(data_path="data/list_of_signals.txt",num_shots=100):

    # Get train shot ids in CSV file
    df = pd.read_csv(data_path.split("data/list_of_signals.txt")[0]+"metadata/2025-05-12/data_splits.csv")

    # Filter rows where the 'train' column is True
    shot_ids = df[df['train'] == True]['shot_id'].tolist() 

    local = True
    store_manager = MASTStorageManager()

    #num_shots = 100 
    signals_average_across_shots_v1(store_manager,
                                shot_ids[:num_shots], 
                                data_path, 
                                local,
                                json_filename="averaged_arrays_N"+str(num_shots)+".json")
    return shot_ids[:num_shots]

def test_conditional_vs_simpler():

    # input
    local = True
    source = "magnetics"
    signal_name = "flux_loop_flux"
    shot_id = 16092

    # MAST store manager
    store_manager = MASTStorageManager()
    shot_ids = store_manager.list_all_shots(local=local)
    shot_ids = shuffle_shot_ids(shot_ids)[:10]

    # Re-order shot_ids so that shot_id is always first
    new_shot_ids = shot_ids[:]
    if shot_id in new_shot_ids:
        shot_ids.remove(shot_id)
        new_shot_ids.insert(0, shot_id)
    else:
        new_shot_ids.insert(0, shot_id)
    shot_ids = new_shot_ids

    # Impute missing channels for test shot_id
    channel_names, conditional_filled = run_conditional_fill(local, source, signal_name, shot_ids, 0)
    conditional_filled = conditional_filled[0].T
    conditional_filled = conditional_filled.to_numpy()
    simple_filled = run_simple_filler(  
        shot_id,
        source,
        signal_name, 
        store_manager,
        local
        )

    # Retrieve original signal
    sig = MASTSignalManager()
    store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})
    vals = sig.get_signal_values(
            data_origin=store,
            source_name=source,
            signal_name=signal_name
        )

    # Plot
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10))

    vmin = min(conditional_filled.min(), simple_filled.min(), vals.min())
    vmax = max(conditional_filled.max(), simple_filled.max(), vals.max())

    p0 = axes[0].pcolorfast(vals)
    p1 = axes[1].pcolorfast(conditional_filled)
    p2 = axes[2].pcolorfast(simple_filled)


    axes[0].set_title(r"$\bf{Original}$" + f" {source}/{signal_name} shot id = {shot_ids[0]}")
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title(r"$\bf{Conditional fill}$")
    fig.colorbar(p1, ax=axes[1])

    axes[2].set_title(r"$\bf{Simple fill}$")
    fig.colorbar(p2, ax=axes[2])

    plt.tight_layout()

    for i in range(3):
        axes[i].set_yticks(np.arange(len(channel_names)))
        axes[i].set_yticklabels(channel_names)

    fig.savefig(f"{source}_{signal_name}.png", bbox_inches="tight", dpi=300)
    plt.show()


def concatenate_signals(
    shot_ids: list[int],
    source_name: str,
    signal_name: str,
    local: bool
    ):
    """_summary_

    Parameters
    ----------
    shot_ids : list[int]
        List of shot IDs
    source_name : str
        Source signal
    signal_name : str
        Signal name
    local : bool
        If True access data locally

    Returns
    -------
    np.ndarray or None
        Concatenated data extracted from shot_ids. 
        
    Original data have shape [n_features, time]. The data are transposed 
    and then concatenated along the features dimension. 
    shot 1:
           feature 1 (time1), feature 2 (time1), feature 3 (time1) .... feature n (time1)
           feature 1 (time2), feature 2 (time2), feature 3 (time2) .... feature n (time2)

    shot 2:
           feature 1 (time1), feature 2 (time1), feature 3 (time1) .... feature n (time1)
           feature 1 (time2), feature 2 (time2), feature 3 (time2) .... feature n (time2)

    .
    .
    .
    shot N:
           feature 1 (time1), feature 2 (time1), feature 3 (time1) .... feature n (time1)
           feature 1 (time2), feature 2 (time2), feature 3 (time2) .... feature n (time2)
    
    """
    # Instanciate MASTStorageManager
    store_manager = MASTStorageManager()
    sig = MASTSignalManager() 

    values = []
    for shot_id in shot_ids:
        store = store_manager.make_shot_store(shot_info={"shot_id":shot_id, "local":local})

        sig_values = sig.get_signal_values(
            data_origin=store,
            source_name=source_name,
            signal_name=signal_name
        )
        if sig_values is not None:
            values.append(sig_values)

    if values:
        return np.concatenate([*values], axis=1).T
    else:
        print("Warning: No values to concatenate.")
        return None
    

def fit_and_save_imputer(
    shot_ids: list[int],
    source_name: str,
    signal_name: str,
    local: bool,
    output_directory: str
    ):

    """Use simple imputer to impute missing data in MAST signals
    
    """
    data = concatenate_signals(shot_ids, source_name, signal_name, local)

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(data)

    joblib.dump(imputer, f"{output_directory}imputer_{signal_name}.joblib")
    print(f"Imputer model saved to {output_directory}imputer_{signal_name}.joblib")


def main(SETTINGS):
    """main funtion to do Imputation of NaN in MAST signals

    Parameters
    ----------
    shot_ids: list[int]
        List containing IDs of shots to process
    SETTINGS: Settings
        Settings object containing configuration parameters
    """
    nr_shots = SETTINGS.GENERAL.nr_shots
    signal_list_file = SETTINGS.LOCALPATHS.signal_list_file
    output_path = SETTINGS.LOCALPATHS.output_path
    data_split_file = SETTINGS.LOCALPATHS.data_split_file
    local = SETTINGS.GENERAL.local
    signal_list =  SETTINGS.SIGFILL.list_of_signals
    
    shot_ids, _, _ = read_data_split_csv(csv_path = data_split_file)
    shot_ids = shuffle_shot_ids(shot_ids)
    
    if nr_shots < len(shot_ids):
        shot_ids = shot_ids[:nr_shots]
    
    store_manager = MASTStorageManager()
    # shotids_not_fully_nan_signals(store_manager,
    #                                 shot_ids,
    #                                 signal_list,
    #                                 local,
    #                                 json_filename=output_path+"signal_shotids_not_fully_nan"+str(len(shot_ids))+".json")

    signals_average_across_shots_v1(store_manager,
                                    shot_ids,
                                    signal_list,
                                    local,
                                    json_filename=output_path+"averaged_arrays_N_"+str(len(shot_ids))+".json")

    for source_signal_name, _ in signal_list:
        source, signal_name = source_signal_name.split("-")
        fit_and_save_imputer(shot_ids, source, signal_name, local, output_path)
        
if __name__ == "__main__":
    
    # Inputs
    SETTINGS = get_settings("scripts/benchmarking/data_processing/config_files/config.json")
    main(SETTINGS)