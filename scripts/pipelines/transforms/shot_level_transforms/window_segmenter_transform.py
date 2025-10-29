"""
window_segmenter_transform.py

This module defines a general-purpose WindowSegmenterTransform class to convert a "shot"
(a dictionary of time series or profile signals) into a list of time-aligned (x, y) windows
for supervised learning (e.g. forecasting, regression, representation learning).

Supports:
---------
- Variable-length segmentation (x and y)
- Asymmetric window sizes (x ≠ y)
- Forecasting with delay (`dt`)
- Stride-based or unitary stepping
- Tolerance for missing/incomplete data
- Consistent signal alignment
- Works with different sampling rates

Input format:
-------------
    shot : dict
        {
            'signal_name': {
                'time': np.ndarray,       # shape (T,)
                'values': np.ndarray      # shape (C, T)
            },
            ...,
        }

Output format:
--------------
    List of windows, each as:
        {
            'x': {
                signal_name: {
                    'time': np.ndarray,   # shape (T_x,)
                    'values': np.ndarray  # shape (C, T_x)
                },
                ...
            },
            'y': {
                signal_name: {
                    'time': np.ndarray,   # shape (T_y,)
                    'values': np.ndarray  # shape (C, T_y)
                },
                ...
            },
            'window_index': int,
        }

Special behaviors:
------------------
- If `stride_unitary = True`: the stride is set to the **maximum** native Δt across all signals.
  This ensures **meaningful movement** for all signals. If `stride_sec` is manually set, then:
      stride = max(stride_sec, max(Δt_i))

- If `x_window_sec` or `y_window_sec` is 0: it defaults to a single sample (i.e., Δt for that signal group).

"""

from typing import List, Dict, Any
import numpy as np


# ======================================================================================================================
class WindowSegmenterTransform:

    """
    Initialize a WindowSegmenterTransform instance.

    This class segments a time-series shot into overlapping (x, y) window pairs for supervised learning.
    It supports signals with different frequencies, asymmetric window sizes, and forecasting delays.

    Parameters
    ----------
    x_keys : list of str
        Names of input signals to use as x in each window.
    y_keys : list of str
        Names of target signals to use as y in each window.
    x_window_sec : float
        Duration (in seconds) of the x-window.
        If set to 0, it will default to the maximum sampling interval among x_keys.
    y_window_sec : float
        Duration (in seconds) of the y-window.
        If set to 0, it will default to the maximum sampling interval among y_keys.
    stride_sec : float or None, optional
        Time difference (in seconds) between the start of consecutive windows.
        If None and `stride_unitary=False`, an error is raised.
        If `stride_unitary=True`, this is ignored.
    dt_sec : float, optional
        Forecasting delay (in seconds) between the end of x-window and start of y-window.
    min_samples_per_window : int, optional
        Minimum number of time points required in each signal window.
        If fewer samples are found, the window is dropped or padded depending on `drop_incomplete_windows`.
    drop_incomplete_windows : bool, optional
        If True, discard windows with insufficient samples in any signal.
        If False, pad such signals with NaNs.
    verbose : bool, optional
        If True, print detailed logs about window segmentation and signal characteristics.
    stride_unitary : bool, optional
        If True, automatically sets the stride to the **maximum Δt** (sampling interval)
        across all signals to ensure all signals move forward in time.
        Useful when signals have different frequencies.

    Notes
    -----
    - Signals must be dictionaries with "time" and "values" fields.
    - Time arrays are assumed to be sorted and monotonic.
    - If both `x_window_sec == 0` and `y_window_sec == 0`, then x_keys and y_keys must have matching sampling rates,
      or a ValueError will be raised.
    - This transform is commonly used as the first step in supervised pipelines for forecasting or sequence modeling.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        x_keys,
        y_keys,
        x_window_sec,
        y_window_sec,
        stride_sec=None,
        dt_sec=0.0,
        min_samples_per_window=1,
        drop_incomplete_windows=False,
        verbose=False,
        stride_unitary=False,
    ):

        self.x_keys = x_keys
        self.y_keys = y_keys
        self.x_window_sec = x_window_sec
        self.y_window_sec = y_window_sec
        self.stride_sec = stride_sec
        self.dt_sec = dt_sec
        self.min_samples = min_samples_per_window
        self.drop_incomplete = drop_incomplete_windows
        self.verbose = verbose
        self.stride_unitary = stride_unitary

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        # Raise Error if some Nans still present in the time vector (fallback but it should have been checked before)
        for var, entry in shot.items():
            if np.isnan(entry["time"]).any():
                raise ValueError(f"[ERROR] Signal '{var}' contains NaN values in its 'time' array.")

        delta_ts = []
        for key in self.x_keys + self.y_keys:
            t = shot[key]["time"]
            dts = np.diff(t)
            if len(dts) > 0:
                delta_ts.append(np.min(dts))
                if self.verbose:
                    print(f"Δt for {key}: {np.min(dts):.6f} s")

        if not delta_ts:
            raise ValueError("No valid Δt found in any signals.")

        min_dt = np.min(delta_ts)
        max_dt = np.max(delta_ts)

        if self.verbose:

            print(f"→ Min Δt across all signals: {min_dt:.6f} s")
            print(f"→ Max Δt across all signals: {max_dt:.6f} s")

        # Compute per-group Δt if needed for x/y
        max_dt_x = max(np.min(np.diff(shot[k]["time"])) for k in self.x_keys)
        max_dt_y = max(np.min(np.diff(shot[k]["time"])) for k in self.y_keys)

        # Handle special case when both windows are zero-length
        if self.x_window_sec == 0 and self.y_window_sec == 0:
            if not np.isclose(max_dt_x, max_dt_y, rtol=1e-3):
                raise ValueError(
                    f"When both x_window_sec and y_window_sec are 0, "
                    f"x_keys and y_keys must have the same sampling rate.\n"
                    f"→ max Δt_x = {max_dt_x:.6f} s\n"
                    f"→ max Δt_y = {max_dt_y:.6f} s"
                )

        # Now handle zero-length x or y windows
        if self.x_window_sec == 0:
            if self.verbose:
                print("[INFO] x_window_sec=0 → using one-sample window with max Δt_x")
            self.x_window_sec = max_dt_x

        if self.y_window_sec == 0:
            if self.verbose:
                print("[INFO] y_window_sec=0 → using one-sample window with max Δt_y")
            self.y_window_sec = max_dt_y

        stride = max_dt if self.stride_unitary else self.stride_sec
        if stride < max_dt:
            if self.verbose:
                print(
                    f"[INFO] Provided stride_sec = {stride:.6f} s is smaller than the maximum sampling interval "
                    f"(Δt = {max_dt:.6f} s) → overriding stride to ensure all signals advance.\n"
                    f"📦 Final stride used: {max_dt:.6f} s"
                )
            stride = max_dt
        else:
            if self.verbose:
                print(f"📦 Final stride used: {stride:.6f} s")

        if stride is None:
            raise ValueError("stride_sec must be set unless stride_unitary=True")

        ref_time = shot[self.x_keys[0]]["time"]
        start_time = ref_time[0]
        end_time = ref_time[-1]

        required_span = self.x_window_sec + self.dt_sec + self.y_window_sec

        if self.verbose:
            print(f"📦 Required total window span: {required_span:.6f} seconds")
            print(f"📦 Using stride: {stride:.6f} seconds")

            print("\n[DEBUG] Signal time ranges and sampling intervals:")
            seen = set()
            for sig in self.x_keys + self.y_keys:
                if sig in seen:
                    continue
                seen.add(sig)
                if sig not in shot:
                    print(f"⚠️  Signal '{sig}' not found in shot")
                    continue
                t = np.asarray(shot[sig]['time'])
                if t.size < 2:
                    print(f"⚠️  Signal '{sig}' has too few samples")
                    continue
                dt = np.min(np.diff(t))
                print(f"  - {sig:30s} spans {t[0]:.5f} → {t[-1]:.5f}, Δt ≈ {dt:.6f}, N = {len(t)}")

            print(f"\n[INFO] Expected x-window duration: {self.x_window_sec:.6f} s")
            print(f"[INFO] Expected y-window duration: {self.y_window_sec:.6f} s")
            print(f"[INFO] Total coverage per window:  {required_span:.6f} s\n")

        results = []
        t_x_start = start_time
        i = 0

        while True:
            t_x_end = t_x_start + self.x_window_sec
            t_y_start = t_x_end + self.dt_sec
            t_y_end = t_y_start + self.y_window_sec

            if t_y_end > end_time:
                break

            x = self._collect_per_signal_windows(shot, self.x_keys, t_x_start, t_x_end)
            y = self._collect_per_signal_windows(shot, self.y_keys, t_y_start, t_y_end)

            if x is not None and y is not None:
                result = {
                    "x": x,
                    "y": y,
                    "window_index": i,
                    # "shot_id": shot.get("shot_id", None)
                }
                results.append(result)

                if self.verbose:
                    x_lengths = {k: v["time"].shape[0] for k, v in x.items()}
                    y_lengths = {k: v["time"].shape[0] for k, v in y.items()}
                    print(
                        f"[Window {i}] "
                        f"x_window: {t_x_start:.5f}–{t_x_end:.5f}, "
                        f"y_window: {t_y_start:.5f}–{t_y_end:.5f}, "
                        f"x_len: {x_lengths}, y_len: {y_lengths}"
                    )

            elif self.verbose:
                print(f"[Window {i}] Skipped (incomplete data)")

            t_x_start += stride
            i += 1

        return results

    # ------------------------------------------------------------------------------------------------------------------
    def _collect_per_signal_windows(self, shot, keys, t_start, t_end):
        signal_slices = {}
        for key in keys:
            signal = shot[key]
            times = signal["time"]
            values = signal["values"]
            mask = (times >= t_start) & (times < t_end)
            idx = np.nonzero(mask)[0]

            # if len(idx) < self.min_samples:
            #     if self.drop_incomplete:
            #         return None
            #     else:
            #         sliced_vals = np.full((values.shape[0], self.min_samples), np.nan)
            #         sliced_time = np.linspace(t_start, t_end, self.min_samples)
            # else:
            
            sliced_vals = values[..., idx[0]:idx[-1]+1]
            sliced_time = times[idx[0]:idx[-1]+1]
            
            if len(idx) >  self.min_samples:
                sliced_vals = values[..., :self.min_samples]
                sliced_time = times[..., :self.min_samples]

            signal_slices[key] = {
                "time": sliced_time,
                "values": sliced_vals
            }

        return signal_slices

    # ------------------------------------------------------------------------------------------------------------------
