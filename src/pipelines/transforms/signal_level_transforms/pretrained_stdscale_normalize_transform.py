import numpy as np
import warnings


# ======================================================================================================================
class StdScalingTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, d):
        """
        Normalize each sample individually: subtract mean, divide by std.

        Input: dict with 'time' and 'values' [features, time]
        Output: same dict, with values normalized per feature
        """

        time = d['time']
        values = d['values']

        if values is not None:
            std_is_zero = ( self.std[..., None] == 0 )
            self.std[..., None][std_is_zero] = 1.0  # avoid division by zero
            values = (values - self.mean[..., None]) / self.std[..., None]
        return {
            'time': time,
            'values': values
        }

    # ------------------------------------------------------------------------------------------------------------------
