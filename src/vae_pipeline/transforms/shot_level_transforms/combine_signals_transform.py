import numpy as np
from typing import Dict, Any, List

class CombineSignalsTransform:
    """
    Combines all signals in `shot` into a single 'combined_signal' and
    returns one sample per time index ('window_index').

    Input (shot)
    ------------
    {
        'signal_name_1': {'time': np.ndarray (T1,), 'values': np.ndarray (C1, T1)},
        'signal_name_2': {'time': np.ndarray (T2,), 'values': np.ndarray (C2, T2)},
        ...
    }

    Output (list of samples)
    ------------------------
    [
        {
            'x': {
                'combined_signal': {'values': np.ndarray (C_tot, 1)}  # column at this time
            },
            'window_index': int,
        },
        ...
    ]

    Notes
    -----
    - This version enforces that `values` are **NumPy arrays** with **ndim=2**.
    - It pads along time (axis=1) with zeros to the max T across signals.
    """

    def __call__(self, shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(shot, dict):
            raise TypeError(f"`shot` must be a dict[str, dict], got {type(shot).__name__}")

        # Collect arrays and validate
        arrays: List[np.ndarray] = []
        lengths: List[int] = []
        dtypes: List[np.dtype] = []

        for name, entry in shot.items():
            if not isinstance(entry, dict):
                raise TypeError(f"shot['{name}'] must be a dict, got {type(entry).__name__}")

            v = entry.get("values", None)
            if v is None:
                raise ValueError(f"shot['{name}']['values'] is None")
 
        
            if v.ndim == 2: 
                arrays.append(v)
                lengths.append(v.shape[1])
                dtypes.append(v.dtype)
                
            if v.ndim ==1:
                arrays.append(v)
                lengths.append(v.size)
                dtypes.append(v.dtype)
            
            if v.ndim>2:
                raise ValueError(f"Current signal shape {v.shape} not supported in CombinedSignalsTransform")
                    
        if not arrays:
            return None

        # Determine output dtype for consistent padding/concat
        out_dtype = np.result_type(*dtypes)

        # Pad each array to max time length
        T_max = max(lengths)
        padded: List[np.ndarray] = []
        for v, T in zip(arrays, lengths):
            if v.dtype != out_dtype:
                v = v.astype(out_dtype, copy=False)
            if T < T_max:
                C = v.shape[0]
                pad = np.zeros((C, T_max - T), dtype=out_dtype)
                v = np.concatenate([v, pad], axis=1)  # pad on time axis
            padded.append(v)

        combined_all = np.concatenate(padded, axis=0) # shape (sum_i C_i, T_max)

        for p in padded:
            print(p.shape)
        # Build one sample per time index
        results: List[Dict[str, Any]] = []
        for t in range(T_max):
            # Keep a time dimension of 1: (C_tot, 1)
            col = combined_all[:, t:t+1]
            results.append({
                "x": {
                    "combined_signal": {
                        "values": col
                    }
                },
                "window_index": int(t),
            })

        return results
