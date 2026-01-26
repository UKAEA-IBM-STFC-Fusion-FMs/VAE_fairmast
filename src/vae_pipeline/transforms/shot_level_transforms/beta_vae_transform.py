import numpy as np
from collections import defaultdict


class BetaVAETransform:
    """
    Transform for β-VAE training - creates signal segments for unsupervised learning.
    """

    def __call__(self, list_samples):
        """
        Transform windowed samples for β-VAE training.

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
        vae_samples : list
            List of (signal_name, signal_data) tuples for β-VAE training
        """
        vae_samples = []
        
        # Loop trhough all window_index
        for sample in list_samples:
            # Extract all signals from both x and y windows
            all_signals = defaultdict(list)

            # Add x signals
            for signal_name, signal_data in sample["x"].items():
                all_signals[signal_name].append(signal_data["values"])

            # Add y signals
            for signal_name, signal_data in sample["y"].items():
                all_signals[signal_name].append(signal_data["values"])

            # Create samples for each unique signal
            for signal_name, signal_segments in all_signals.items():
                for segment in signal_segments:
                    # Flatten multi-channel signals or keep 1D signals as is
                    if segment.ndim > 1:
                        for channel_idx in range(segment.shape[0]):
                            vae_samples.append(
                                {
                                    "signal_name": f"{signal_name}_ch{channel_idx}",
                                    "data": segment[channel_idx, :].astype(np.float32),
                                    "original_signal": signal_name,
                                    "channel": channel_idx,
                                    "window_index": sample.get("window_index", 0),
                                }
                            )
                    else:
                        vae_samples.append(
                            {
                                "signal_name": signal_name,
                                "data": segment.astype(np.float32),
                                "original_signal": signal_name,
                                "channel": 0,
                                "window_index": sample.get("window_index", 0),
                            }
                        )

        return vae_samples