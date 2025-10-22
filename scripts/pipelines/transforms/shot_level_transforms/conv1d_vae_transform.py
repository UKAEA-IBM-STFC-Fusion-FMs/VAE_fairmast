import numpy as np
from collections import defaultdict
import torch

class Conv1dVAETransform:
    """
    Transform for conv1d-VAE training - creates signal segments for unsupervised learning.
    """

    def __call__(self, list_samples):
        """
        Transform windowed samples.

        Parameters
        ----------
        list_samples : list
        [
            {
                'x': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.ndarray  # shape (C, T_x)
                    },
                    signal_name2: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.ndarray  # shape (C, T_x)
                    },
                    …..
                },
                'y': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_y,)
                        'values': np.ndarray  # shape (C, T_y)
                    },
                    signal_name2: {
                                'time': np.ndarray,   # shape (T_y,)
                                'values': np.ndarray  # shape (C, T_y)
                    },
                    ...
                },
            'window_index': int,  # The same index for x, y does not mean necessarily same time, it means that x and y are to be considered input-target pair
            }
            ... Same for a different 'window_index'
        ]

        Returns
        -------
        all_signals : dict
            "signal_name": [signal_values_idx_1, ..., signal_values_idx_n].
            The list contains tensors, one for each temporal window
        """

        all_signals = defaultdict(list)
        window_ids = []
        # Loop trhough all window_index
        if not list_samples or len(list_samples)==0:
            print("Empty list_samples in conv1d_vae_transform.py")
            return all_signals
        for windowed_signal in list_samples:
            
            window_id = windowed_signal['window_index']
            if window_id not in  window_ids:
                 window_ids.append(window_id)
            
            # Add signals
            for signal_name, signal_data in windowed_signal["x"].items():
                all_signals[signal_name].append(torch.tensor(signal_data["values"],dtype=torch.float32))

        # Sanity check to be removed later
        for name, val in all_signals.items():
            if len(window_ids) != len(val):
                print(f"Error, len(window_ids) != len(val): {len(window_ids)} != {len(val)}")
        for nr, w in enumerate(window_ids):
            if nr != int(w):
                print("Error, non sequential window_ids")

        return all_signals

        