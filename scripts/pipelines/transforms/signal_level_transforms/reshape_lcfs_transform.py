import numpy as np
from scipy.ndimage import zoom

# ======================================================================================================================
class ReshapeLcfsTransform:

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_):
        """
        Input: torch dict with key 'time' and key 'values'.

        Returns: torch dict with key 'time' and key 'values with NaNs of profiles (i.e., when one full channel in the
        profile is missing) filled with zeros.
        """

        time = dict_['time']
        values = dict_['values']

        new_profile = []

        for vector in values.T:
            cleaned_vector = vector[~np.isnan(vector)]
            V = len(cleaned_vector)
            if V == 0:
                resized_vector = np.full((170,), np.nan)
            else:
                scale_factor = 170 / V
                resized_vector = zoom(cleaned_vector, (scale_factor), order=1)  # order=1 = linear interpolation
            new_profile.append(resized_vector)

        return {
            'time': time,
            'values': np.stack(new_profile).T
        }

    # ------------------------------------------------------------------------------------------------------------------