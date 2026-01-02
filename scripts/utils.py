import numpy as np
import os
import pandas as pd
import random
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)

from MAST_tools.signal_utils import MASTSignalManager  # noqa
from MAST_tools.store_utils import MASTStorageManager  # noqa


# ----------------------------------------------------------------------------------------------------------------------
def is_finite_numeric_array(arr):
    # Check if array is numeric
    if not np.issubdtype(arr.dtype, np.number):
        return False
    # Check if all values are finite (no NaN or Inf)
    return np.isfinite(arr).all()


# ----------------------------------------------------------------------------------------------------------------------
def read_signals(filepath: str) -> dict[str:int]:
    """
    filepath: path to file containing all signal names and their multiplicity (nr. of channels)

    Output: Dictionary with signal names as keys and number of traces as values. Empty dictionary if exception raised
            during reading.
    """

    signals = dict()
    try:
        with open(filepath, "r") as file:
            for line in file:
                signals[line.split()[0]] = int(line.split()[1])
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file '{filepath}': {e}")
        return {}
    return signals


# ----------------------------------------------------------------------------------------------------------------------
def shuffle_shot_ids(shot_ids, seed=None):
    random.seed(seed)
    random.shuffle(shot_ids)
    return shot_ids


# ----------------------------------------------------------------------------------------------------------------------
def make_dataframe_from_shot_ids(
        store_manager: MASTStorageManager,
        shot_ids: list[int],
        group: str,
        signal_name: str,
        local=True
):
    """Return a dataFrame from concatenated signals"""

    channels = None
    shot_list = []
    for shot_id in shot_ids:
        
        store = store_manager.make_shot_store(shot_info={"shot_id": shot_id, "local": local})
        sig = MASTSignalManager()

        sig_values = sig.get_signal_values(
            data_origin=store,
            source_name=group,
            signal_name=signal_name
        )

        if channels is None:
            channels = list(sig.get_channel_names(store, group, signal_name))

        df = pd.DataFrame(sig_values.T, columns=channels)
        shot_list.append(df)
    
    return channels, shot_list


# ======================================================================================================================
if __name__ == "__main__":
    print("Main")
