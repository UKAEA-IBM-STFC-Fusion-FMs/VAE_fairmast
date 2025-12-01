import numpy as np
from collections import defaultdict
# ======================================================================================================================
class CNNTransform:

    def __init__(self):
        # dictionary that persists across calls
        self.var_groups = {"x": None, "y": None}

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, list_samples):
        cnn_samples = []

        for sample in list_samples:
            # keep variable names
            array_x = [data['values'].T for var, data in sample['x'].items()]
            names_x = list(sample['x'].keys())
            array_y = [data['values'].T for var, data in sample['y'].items()]
            names_y = list(sample['y'].keys())

            # group arrays by shape for CNN branch
            reshaped_x, groups_x = self._group_arrays_by_shape(array_x, names_x)
            reshaped_x = [arr.squeeze(0) if arr.shape[0] == 1 else arr for arr in reshaped_x]
            reshaped_x = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_x]
            reshaped_y, groups_y = self._group_arrays_by_shape(array_y, names_y)
            reshaped_y = [arr.squeeze(0) for arr in reshaped_y]
            reshaped_y = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in reshaped_y]

            # update mappings
            self._update_groups("x", groups_x)
            self._update_groups("y", groups_y)

            cnn_samples.append((reshaped_x, reshaped_y))
        
        return cnn_samples

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _group_arrays_by_shape(arrays, names):
        grouped = defaultdict(lambda: {"arrays": [], "names": []})
        for arr, name in zip(arrays, names):
            grouped[arr.shape]["arrays"].append(arr)
            grouped[arr.shape]["names"].append(name)

        reshaped_list, name_groups = [], []
        for same_shape_group in grouped.values():
            stacked = np.stack(same_shape_group["arrays"])
            transposed = np.transpose(stacked, axes=[1, 0] + list(range(2, stacked.ndim)))
            reshaped_list.append(transposed)
            name_groups.append(same_shape_group["names"])

        return reshaped_list, name_groups

    # ------------------------------------------------------------------------------------------------------------------
    def _update_groups(self, kind, groups):
        """Store mapping of variable groups only once."""
        if self.var_groups[kind] is None:  # only set the first time
            self.var_groups[kind] = groups

