import numpy as np
from collections import defaultdict
import torch
from typing import List, Dict, Any

class Cast1DTransform:
    """
        Transform windowed samples by flatening "values" into a 1D array.
        values.shape = (Nr_F ,T_x) --> values.shape = (Nr_F * T_x)

        Parameters
        ----------
        list_samples : list of time windows determined by 'window_index'
        [
            {
                'x': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.ndarray  # shape (Nr_F, T_x) 
                    },
             
                'window_index': int,  # The same index for x, y does not mean necessarily same time, it means that x and y are to be considered input-target pair
            },
            ... Same for a different 'window_index'
        ]
       
        
        Returns
        -------
         list_samples : list of time windows determined by 'window_index.
          [
            {
                'x': {
                    signal_name1: {
                        'time': np.ndarray,   # shape (T_x,)
                        'values': np.array  # shape (Nr_F * T_x) 
                    },
             
                'window_index': int,  # The same index for x, y does not mean necessarily same time, it means that x and y are to be considered input-target pair
            },
            ... Same for a different 'window_index'
        ]
        ]
            
        """

    def __call__(self, list_samples:List[Dict[str, Any]]):
       
        new_list_samples: List[Dict[str, Any]] = []
        
        # Loop trhough all window_index
        if not list_samples:
            return None
        
        for windowed_signal in list_samples:
            for signal_name, signal_data in windowed_signal["x"].items():
                
                if len(signal_data["values"].shape) > 1:
                    windowed_signal["x"][signal_name]["values"] =  signal_data["values"].reshape(-1)
                
                new_list_samples.append(windowed_signal)

        if new_list_samples:
            return new_list_samples
        else:
            return None

        