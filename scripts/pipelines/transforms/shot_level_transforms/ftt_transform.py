from typing import List, Dict, Any, Tuple
import numpy as np


# ======================================================================================================================
class FTTransformPrep:
    """
    Prepare segmented windows into FTTransformer-ready tuples.

    Input format (from WindowSegmenterTransform):
      shot -> List[window], where each window is:
        {
          "x": { x_key: {"values": np.ndarray, ...}, ... },
          "y": { y_key: {"values": np.ndarray, ...}, ... },
          ... (other metadata)
        }

    Output per window (for collate_fttransform):
        (Xs, names, y_native)
          Xs       : List[np.ndarray] of inputs (now optionally normalized to 3D)
          names    : List[str] target names (order preserved)
          y_native : Dict[target_name] -> np.ndarray, normalized to 3D if requested

    Normalization to 3D (applied when enabled):
      - 1D  (T,)         -> (1, 1, T)
      - 2D  (C, T)       -> (C, 1, T)
      - 2D  (D, 1)       -> (D, 1, 1)
      - 3D  (C, T, 1)    -> (C, 1, T)   (assume axis1 is time, axis2 is singleton)
      - 3D  (C, 1, T)    -> (C, 1, T)   (already canonical for 1D-over-time)
      - Image/Video (H, W, T) or (H, W, 1) are left as-is (d1>1 and d2>1)

    Parameters
    ----------
    x_keys : List[str]
        List of signal names to be used as inputs (in order).
    y_key_to_target : Dict[str, str]
        Mapping from window['y'] keys to model target names.
        (If you later add a version that accepts `y_keys`, you can build {k:k} upstream.)
    ensure_3d_inputs : bool
        If True, normalize X arrays to 3D as described above.
    ensure_3d_targets : bool
        If True, normalize Y arrays to 3D as described above.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        x_keys: List[str],
        y_key_to_target: Dict[str, str],
        ensure_3d_inputs: bool = True,
        ensure_3d_targets: bool = True,
    ):
        self.x_keys = x_keys
        self.y_key_to_target = y_key_to_target
        self.ensure_3d_inputs = ensure_3d_inputs
        self.ensure_3d_targets = ensure_3d_targets

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _to_3d(arr: np.ndarray) -> np.ndarray:
        """Normalize array to (d1, d2, d3) with sensible rules for TS/profile/scalar."""

        a = np.asarray(arr)
        if a.ndim == 1:  # noqa
            # (T,) -> (1,1,T)
            return a[None, None, :]
        if a.ndim == 2:  # noqa
            d1, d2 = a.shape  # noqa
            if d2 > 1:
                # (C,T) -> (C,1,T)
                return a.reshape(d1, 1, d2)  # noqa
            else:
                # (D,1) -> (D,1,1)
                return a.reshape(d1, 1, 1)  # noqa
        if a.ndim == 3:  # noqa
            d1, d2, d3 = a.shape  # noqa
            # Treat strictly 2D spatial (image/video) as already canonical
            if d1 > 1 and d2 > 1:
                return a
            # If it looks like (C,T,1), convert to (C,1,T)
            if d3 == 1 and d2 > 1:
                return np.transpose(a, (0, 2, 1))  # (C,1,T)
            # (C,1,T) already fine, or (D,1,1) scalar/vector-like
            return a

        # Fallback: add singleton dims until 3D
        while a.ndim < 3:
            a = a[..., None]

        return a

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> List[Tuple[List[np.ndarray], List[str], Dict[str, np.ndarray]]]:

        if not isinstance(shot, list):
            raise ValueError("FTTransformPrep expects shot-level transform input to be a list of windows.")
            # ⬇️ Skip immediately if previous transforms dropped all windows
        if len(shot) == 0:
            print("Shot is empty, returning empty list.")
            return []  # Do not preprocess empty shots

        samples = []
        for window in shot:

            # ..........................................................................................................
            # 1) Inputs in specified order

            Xs: List[np.ndarray] = []
            for key in self.x_keys:
                if key not in window["x"]:
                    raise KeyError(f"Missing expected x_key '{key}' in window.")
                arr = np.asarray(window["x"][key]["values"])
                if self.ensure_3d_inputs:
                    arr = self._to_3d(arr)
                Xs.append(arr)

            # ..........................................................................................................
            # 2) Targets (map y_key -> target name)

            y_native: Dict[str, np.ndarray] = {}
            names: List[str] = []
            for y_key, target_name in self.y_key_to_target.items():
                if y_key not in window["y"]:
                    raise KeyError(f"Missing expected y_key '{y_key}' in window.")
                arr = np.asarray(window["y"][y_key]["values"])
                if self.ensure_3d_targets:
                    arr = self._to_3d(arr)
                y_native[target_name] = arr
                names.append(target_name)

            samples.append((Xs, names, y_native))

            # ..........................................................................................................

        return samples

    # ------------------------------------------------------------------------------------------------------------------
