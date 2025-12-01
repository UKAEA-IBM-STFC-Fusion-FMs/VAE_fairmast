import numpy as np


# ======================================================================================================================
class SamplingToReferenceTimeTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, ref_freq):
        self.ref_freq = ref_freq

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: torch dict with key 'time' and key 'values'
        Returns: torch dict with key 'time' and key 'values with NaNs forward-filled
        """

        time = dict_['time']
        values = dict_['values']
        # print('\nAfter time rescaling', values.shape[-1] )
        ref_time = []
        for k in range(0, int((time[-1]-time[0])/self.ref_freq) + 2):
            ref_time.append(time[0] + k*self.ref_freq)

        # Up-sample or subsample by picking the closest value in time to the ref_time
        idx = np.searchsorted(time, ref_time)  # Reformat to be on the same scale as ref_time
        idx = np.clip(idx, 1, len(time) - 1)
        # print(values.shape)
        values_rescaled = values[..., idx]
        # print('\nAfter time rescaling', values_rescaled.shape[-1] )

        return {
            'time': ref_time,
            'values': values_rescaled
        }

    # ------------------------------------------------------------------------------------------------------------------
