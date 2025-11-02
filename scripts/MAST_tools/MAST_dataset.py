import numpy as np
from torch.utils.data import Dataset
from .signal_utils import MASTSignalManager

# ======================================================================================================================
class MastDataset(Dataset):
    """Dataset class for MAST data.

    MAST Dataset extending the base torch.utils.data.Dataset class. See __init__ below for details.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            local: bool,
            shots_list: list[int],
            source_signal_list: list[str],
            signal_level_transform_map=None,
            shot_level_transform=None
    ):
        """ Initialize the MASTDataset.
        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket
        shots_list : list[int]
            List of shot IDs to load data for.
        source_signal_list : list[str]
            List of data names to load, format: ('source', 'signal')
        signal_level_transform_map : _type_, optional
            dict map of pipeline of transforms to apply at signal level, by default None
        shot_level_transform : _type_, optional
            pipeline of transforms to apply at shot level, by default None
        """

        self.local = local
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.signal_level_transform_map = signal_level_transform_map
        self.shot_level_transform = shot_level_transform
        self.sig = MASTSignalManager()

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.shots_list)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):

        store_manager = self.sig.store_manager
        store = store_manager.make_shot_store(
            shot_info={
                "shot_id": self.shots_list[idx],
                "local": self.local  # Add among attributes
                }
            )

        shot = {}

        # Collect variables (i.e. source-signal) of interest
        for source, signal in self.source_signal_list:

            shot_profile = self.sig.get_signal_profile(
                    data_origin=store,
                    source_name=source,
                    signal_name=signal
                    )
            
            if shot_profile is not None:
                try:
                    shot_time, _ = self.sig.get_signal_times_and_time_type(
                                signal,
                                store,
                                source
                                )
                except Exception as e:
                    print(f"Error getting time for shot {self.shots_list[idx]}: {e}")
                    shot_time = None
                try:
                    if f"{source}-{signal}" == "equilibrium-psi":
                        print("Transposing values because equilibrium-psi not saved the same way as others")
                        # print(shot_profile.values.shape)
                        shot_vals = np.moveaxis(shot_profile.values, 0, -1)
                        # print(shot_vals.shape)
                    else:
                        shot_vals = (np.expand_dims(shot_profile.values, axis=0) if shot_profile.values.ndim == 1
                                    else shot_profile.values)
                        # print(shot_vals.shape)

                except AttributeError:
                    shot_vals = None
            else:
                shot_vals = None
                shot_time = None
            
            # Apply variable-level transforms
            if self.signal_level_transform_map and (shot_vals is not None and shot_time is not None):
                shot[f'{source}-{signal}'] = self.signal_level_transform_map[f'{source}-{signal}'](
                    {"time": shot_time, "values": shot_vals}
                )
            else:
                shot[f'{source}-{signal}'] = {"time": shot_time, "values": shot_vals}

        # Apply shot-level transforms to obtain a list of training objects
        if self.shot_level_transform:
            if all(subval is not None
                   for subdict in shot.values()
                   for subval in subdict.values()):
                list_chunks = self.shot_level_transform(shot)
                return list_chunks
            else:
                print('Nan still present in shot')
                return []
        else:
            return shot

    # ------------------------------------------------------------------------------------------------------------------


class CachedDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: any torch Dataset (e.g. torchvision, custom, etc.)
        """
        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        self._is_cached = [False] * len(base_dataset)
    
    def __getitem__(self, idx):
        # If not cached, load it once
        if not self._is_cached[idx]:
            item = self.base_dataset[idx]

            # # Optional: pin to memory for faster GPU transfer
            # if isinstance(item, torch.Tensor):
            #     item = item.pin_memory()
            # elif isinstance(item, (list, tuple)):
            #     item = tuple(x.pin_memory() if isinstance(x, torch.Tensor) else x for x in item)

            self.cache[idx] = item
            self._is_cached[idx] = True

        return self.cache[idx]

    def __len__(self):
        return len(self.base_dataset)