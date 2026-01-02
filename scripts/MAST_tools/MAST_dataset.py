import numpy as np
from torch.utils.data import Dataset
from .signal_utils import MASTSignalManager

# ======================================================================================================================
class MastDataset(Dataset):
    """Dataset class for MAST data.

    If `return_incomplete_shot` is True, __getitem__ returns shots/windows even when
    some variables are missing (time/values is None). This lets downstream windowing
    mark missing inputs per-sample with `None`.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        local: bool,
        shots_list: list[int],
        source_signal_list: list[str],
        signal_level_transform_map=None,
        shot_level_transform=None,
        return_incomplete_shots: bool = False,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        local : bool
            If True, use local MAST database, otherwise use remote S3 bucket
        shots_list : list[int]
            List of shot IDs to load data for.
        source_signal_list : list[str]
            List of data names to load, format: ('source', 'signal')
        signal_level_transform_map : dict[str, Callable], optional
            Map of transforms to apply at signal level.
        shot_level_transform : Callable, optional
            Transform to apply at shot level; typically builds windows.
        return_incomplete_shot : bool
            If True, do NOT drop shots with missing variables; pass them through so that
            the windowing transform can insert `None` for missing inputs per window.
        """
        self.local = local
        self.shots_list = shots_list
        self.source_signal_list = source_signal_list
        self.signal_level_transform_map = signal_level_transform_map
        self.shot_level_transform = shot_level_transform
        self.return_incomplete_shots = return_incomplete_shots  # NEW
        self.sig = MASTSignalManager()

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.shots_list)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        store_manager = self.sig.store_manager
        store = store_manager.make_shot_store(
            shot_info={"shot_id": self.shots_list[idx], "local": self.local}
        )

        shot = {}

        # Collect variables (i.e. source-signal) of interest
        for source, signal in self.source_signal_list:
            shot_profile = self.sig.get_signal_profile(
                data_origin=store, source_name=source, signal_name=signal
            )

            if shot_profile is not None:
                try:
                    shot_time, _ = self.sig.get_signal_times_and_time_type(signal, store, source)
                except Exception as e:
                    print(f"Error getting time for shot {self.shots_list[idx]}: {e}")
                    shot_time = None
                try:
                    shot_vals = (
                        np.expand_dims(shot_profile.values, axis=0)
                        if shot_profile.values.ndim == 1
                        else shot_profile.values
                    )
                except AttributeError:
                    shot_vals = None
            else:
                shot_vals = None
                shot_time = None

            # Apply variable-level transforms only if we have both time and values
            if self.signal_level_transform_map and (shot_vals is not None and shot_time is not None):
                shot[f"{source}-{signal}"] = self.signal_level_transform_map[f"{source}-{signal}"](
                    {"time": shot_time, "values": shot_vals}
                )
            else:
                # Keep missing signals as {"time": None, "values": None}
                shot[f"{source}-{signal}"] = {"time": shot_time, "values": shot_vals}

        # Apply shot-level transforms to obtain a list of training objects (windows)
        if self.shot_level_transform:
            if self.return_incomplete_shots:
                # Pass through even if some variables are missing; the windower will insert None per window.
                list_chunks = self.shot_level_transform(shot)
                return list_chunks if isinstance(list_chunks, list) else [list_chunks]
            else:
                # Legacy behavior: drop shots with any missing variable
                if all(subval is not None for subdict in shot.values() for subval in subdict.values()):
                    list_chunks = self.shot_level_transform(shot)
                    return list_chunks if isinstance(list_chunks, list) else [list_chunks]
                else:
                    return []
        else:
            # No shot-level transform → return the raw shot dict (may include None fields)
            return shot

    # ------------------------------------------------------------------------------------------------------------------
    # Minimal additions for caching/baking support
    # ------------------------------------------------------------------------------------------------------------------
    def get_shot_id(self, idx: int):
        return self.shots_list[idx]

    def get_windows_for_shot(self, idx: int):
        """Return the list of windows for the given shot index ([] if empty/missing)."""
        try:
            obj = self.__getitem__(idx)
        except Exception:
            return []
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            # No shot-level transform → treat as a single-window shot
            return [obj]
        else:
            return []

    def get_window_index_within_shot(self, idx: int):
        return None

class CachedDataset(Dataset):
    def __init__(self, base_dataset):
        """
        base_dataset: torch Dataset
        """
        self.base_dataset = base_dataset
        self.cache = [None] * len(base_dataset)
        self._is_cached = [False] * len(base_dataset)
    
    def __getitem__(self, idx):
        # If not cached, load it once
        if not self._is_cached[idx]:
            item = self.base_dataset[idx]

            self.cache[idx] = item
            self._is_cached[idx] = True

        return self.cache[idx]

    def __len__(self):
        return len(self.base_dataset)
