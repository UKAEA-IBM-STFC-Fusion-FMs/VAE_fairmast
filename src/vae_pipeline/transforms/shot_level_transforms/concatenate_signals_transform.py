from typing import List, Dict, Any
import numpy as np

class ConcatenateSignalsBeforeTimeSegmentation():
    """
    Concatenate shot signals along the feature axis.
    Expects each entry in `shot` to have:
      - "time": 1D ndarray of length T
      - "values": ndarray of shape (F_i, T)
    Returns a dict with:
      - "time": 1D ndarray of length T
      - "values": ndarray of shape (sum_i F_i, T)
    """

    
    def __call__(self, shot: Dict[str, Any]):
       
        # Retrieve "values" and "time" for each signal in shot
        times = []
        values = []

        for signal_name, entry in shot.items():
            
            t = entry["time"]
            v = entry["values"]

            if not isinstance(t, np.ndarray) or t.ndim != 1:
                raise ValueError(f"[ERROR] Signal '{signal_name}' 'time' must be a 1D np.ndarray.")
            if np.isnan(t).any():
                raise ValueError(f"[ERROR] Signal '{signal_name}' contains NaN in its 'time' array.")
            if not isinstance(v, np.ndarray):
                raise ValueError(f"[ERROR] Signal '{signal_name}' 'values' must be an np.ndarray.") 
            if v.shape[1] != t.shape[0]:
                raise ValueError(
                    f"[ERROR] Signal '{signal_name}' values.shape[1] must match len(time). "
                    f"Got values.shape[1]={v.shape[1]} vs len(time)={t.shape[0]}."
                )

            times.append(t)
            values.append(v)
        
        
        # Verify all time arrays are identical. Use allclose for float safety.
        ref_time = times[0]
        for i, t in enumerate(times[1:], start=1):
            if t.shape != ref_time.shape or not np.allclose(t, ref_time, equal_nan=False):
                raise ValueError(
                    f"[ERROR] Times differ between signals 0 and {i}. "
                    "Concatenation requires identical time vectors."
                )

        # Concatenate along feature axis.
        values_cat = np.concatenate(values, axis=0)
        
        # New shot, first signal_name is used to store concatenated values.
        first_key = next(iter(shot.keys()))
        shot = {
                first_key: 
                    {
                        'time': times[0],
                        'values': values_cat    
                    }
            }

        return shot

class ConcatenateSignalsAfterTimeSegmentation():
    """
        Concatenate windowed samples along features axis.

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
    Expects each entry in `shot` to have:
      - "time": 1D ndarray of length T
      - "values": ndarray of shape (F_i, T)
    Returns a list of samples (dict) 
        - each sample is a dict with a single 'x' entry. This has a single signal place holder obtained from the last signal in the list of signals:
        - "time": 1D ndarray of length T
        - "values": ndarray of shape (sum_i F_i, T)
    """ 
    def __call__(self, list_samples, signal_place_holder=None):

        new_list_samples = []
        
        # Loop trhough all window_index
        if not list_samples:
            return None
    
        signal_place_holder = None
        for nr, windowed_signal in enumerate(list_samples):
            
            times = []
            values = []
            
            # Loop through all signals
            for signal_name, entry in windowed_signal["x"].items():
                
                t = entry["time"]
                v = entry["values"]

                if not isinstance(t, np.ndarray) or t.ndim != 1:
                    raise ValueError(f"[ERROR] Signal '{signal_name}' 'time' must be a 1D np.ndarray.")
                
                if np.isnan(t).any():
                    raise ValueError(f"[ERROR] Signal '{signal_name}' contains NaN in its 'time' array.")
                
                if not isinstance(v, np.ndarray):
                    raise ValueError(f"[ERROR] Signal '{signal_name}' 'values' must be an np.ndarray.") 
                
                if v.shape[1] != t.shape[0]:
                    raise ValueError(
                        f"[ERROR] Signal '{signal_name}' values.shape[1] must match len(time). "
                        f"Got values.shape[1]={v.shape[1]} vs len(time)={t.shape[0]}."
                    )

                times.append(t)
                values.append(v)
                
                if signal_place_holder is None:
                    signal_place_holder = signal_name
                
            # Verify all time arrays are identical
            ref_time = times[0]
            for i, t in enumerate(times[1:], start=1):
                if t.shape != ref_time.shape or not np.allclose(t, ref_time, equal_nan=False):
                    raise ValueError(
                        f"[ERROR] Times differ between signals 0 and {i}. "
                        "Concatenation requires identical time vectors."
                    )

            # Concatenate along feature axis.
            values_cat = np.concatenate(values, axis=0)
            
            # New sample, first signal_name is used to store concatenated values.
            sample = {
                'x':{
                    signal_place_holder :
                        {
                            'time': times[0],
                            'values': values_cat    
                        }        
                },
                'window_index':nr
            }
            new_list_samples.append(sample)

        if new_list_samples:
            return new_list_samples
        else:
            return None
            
            
                
                
