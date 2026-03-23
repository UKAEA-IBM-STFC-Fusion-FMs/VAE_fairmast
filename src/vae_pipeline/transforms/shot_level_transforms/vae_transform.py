import numpy as np
from collections import defaultdict
import torch

class VAETransform:
    """
        Transform windowed samples into a dictionary.

        Parameters
        ----------
        list_samples : list of time windows determined by 'window_index'
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
            },
            ... Same for a different 'window_index'
        ]
        targeted_signal_length: int
        
        HINTS: 
        - Windowed signals are cropped to this length if longer. 
        - Windowed signals are disregarded is shorter than targeted_signal_length. Thisn occurrence should be rare.
        - The user must make sure targeted_signal_length is the minimum signal length among all the windowed signals in list_samples.
        - Targeted_signal_length is defined by rounding down the x_window_sec/(signal frequency)
        
        Returns
        -------
        all_signals : dict
            "signal_name": [signal_values_idx_1, ..., signal_values_idx_n] for 1,...,n temporal windows
            
        """
        
    def __init__(self, targeted_signal_length:int):
        self.targeted_signal_length = targeted_signal_length

    def __call__(self, list_samples):
        all_signals = defaultdict(list)

        # Loop trhough window_index
        if not list_samples or list_samples is None:
            return None

        for windowed_signal in list_samples:
            for signal_name, signal_data in windowed_signal["x"].items():
               
                values = signal_data["values"]
                
                if values.shape[-1] < self.targeted_signal_length:
                    continue
                
                # Slice last dimension (or keep as is)
                cropped_values = values[..., :self.targeted_signal_length]
                      
                all_signals[signal_name].append(torch.tensor(cropped_values, dtype=torch.float32))

        if all_signals:
            return all_signals
        else:
            return None

        