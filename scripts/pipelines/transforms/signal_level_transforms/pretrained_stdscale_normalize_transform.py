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
    
    
class StdScalingTransform_v2:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean, std):
        self.mean = np.mean(mean)
        self.std = np.sqrt(np.sum(std**2))/len(mean)

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
           
            values = (values - self.mean) / self.std
        return {
            'time': time,
            'values': values
        }