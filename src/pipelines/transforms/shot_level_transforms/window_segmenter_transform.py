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

from typing import List, Dict, Any, Optional
import numpy as np


# ======================================================================================================================
class WindowSegmenterTransform:
    def __init__(
        self,
        x_keys,
        y_keys,
        x_window_sec,
        y_window_sec,
        stride_sec=None,
        dt_sec=0.0,
        min_samples_per_window=1,
        drop_incomplete_windows=True,
        verbose=False,
        stride_unitary=False,
    ):
        self.x_keys = list(x_keys)
        self.y_keys = list(y_keys)
        self.x_window_sec = float(x_window_sec)
        self.y_window_sec = float(y_window_sec)
        self.stride_sec = None if stride_sec is None else float(stride_sec)
        self.dt_sec = float(dt_sec)
        self.min_samples = int(min_samples_per_window)
        self.drop_incomplete = bool(drop_incomplete_windows)
        self.verbose = bool(verbose)
        self.stride_unitary = bool(stride_unitary)

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
    def __call__(self, shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Guard only on available time arrays (and entries that are dicts)
        for var, entry in shot.items():
            if isinstance(entry, dict):
                t = entry.get("time", None)
                if t is not None and np.isnan(t).any():
                    raise ValueError(f"[ERROR] Signal '{var}' contains NaN values in its 'time' array.")

        # --- helpers over the available signals ---
        def _min_dt_of(sig_name: str) -> Optional[float]:
            d = shot.get(sig_name, None)
            if not isinstance(d, dict):
                return None
            t = d.get("time", None)
            if t is None or len(t) < 2:
                return None
            dts = np.diff(np.asarray(t))
            pos = dts[dts > 0]
            return float(np.min(pos)) if pos.size else None

        def _group_max_dt(keys: list[str]) -> Optional[float]:
            vals = [dt for k in keys if (dt := _min_dt_of(k)) is not None]
            return float(np.max(vals)) if vals else None

        # Δt over all x+y where available
        delta_ts = []
        for key in self.x_keys + self.y_keys:
            dt = _min_dt_of(key)
            if dt is not None:
                delta_ts.append(dt)
                if self.verbose:
                    print(f"Δt for {key}: {dt:.6f} s")
        if not delta_ts:
            raise ValueError("No valid Δt found in any available signals.")

        min_dt = float(np.min(delta_ts))
        max_dt = float(np.max(delta_ts))
        if self.verbose:
            print(f"→ Min Δt across available signals: {min_dt:.6f} s")
            print(f"→ Max Δt across available signals: {max_dt:.6f} s")

        # Per-group Δt (skip missing)
        max_dt_x = _group_max_dt(self.x_keys)
        max_dt_y = _group_max_dt(self.y_keys)

        # Use local window sizes (don't mutate self.*)
        x_win_sec = float(self.x_window_sec)
        y_win_sec = float(self.y_window_sec)

        # Special case when both windows are zero-length
        if x_win_sec == 0 and y_win_sec == 0:
            if max_dt_x is None or max_dt_y is None:
                raise ValueError("x_window_sec=y_window_sec=0 requires at least one present signal in each group.")
            if not np.isclose(max_dt_x, max_dt_y, rtol=1e-3):
                raise ValueError(
                    "When both x_window_sec and y_window_sec are 0, x_keys and y_keys must have the same sampling rate.\n"
                    f"→ max Δt_x = {max_dt_x:.6f} s\n"
                    f"→ max Δt_y = {max_dt_y:.6f} s"
                )

        # Zero-length x or y: promote to one-sample window using group Δt
        if x_win_sec == 0:
            if max_dt_x is None:
                raise ValueError("x_window_sec=0 but no available x signal to infer Δt.")
            if self.verbose:
                print("[INFO] x_window_sec=0 → using one-sample window with max Δt_x")
            x_win_sec = max_dt_x

        if y_win_sec == 0:
            if max_dt_y is None:
                raise ValueError("y_window_sec=0 but no available y signal to infer Δt.")
            if self.verbose:
                print("[INFO] y_window_sec=0 → using one-sample window with max Δt_y")
            y_win_sec = max_dt_y

        # Stride
        stride = max_dt if self.stride_unitary else self.stride_sec
        if stride is None:
            raise ValueError("stride_sec must be set unless stride_unitary=True")
        if stride < max_dt:
            if self.verbose:
                print(
                    f"[INFO] Provided stride_sec = {float(self.stride_sec):.6f} s is smaller than the maximum sampling interval "
                    f"(Δt = {max_dt:.6f} s) → overriding stride to ensure all signals advance.\n"
                    f"📦 Final stride used: {max_dt:.6f} s"
                )
            stride = max_dt
        else:
            if self.verbose:
                print(f"📦 Final stride used: {stride:.6f} s")

        # Choose a reference time among available x_keys
        ref_time = None
        for k in self.x_keys:
            d = shot.get(k, None)
            if isinstance(d, dict):
                t = d.get("time", None)
                if t is not None and len(t) > 0:
                    ref_time = np.asarray(t)
                    break
        if ref_time is None:
            raise ValueError("No available x signal with time array to drive window stepping.")

        start_time = float(ref_time[0])
        end_time = float(ref_time[-1])

        required_span = x_win_sec + self.dt_sec + y_win_sec
        if self.verbose:
            print(f"📦 Required total window span: {required_span:.6f} seconds")
            print(f"📦 Using stride: {stride:.6f} seconds")

            print("\n[DEBUG] Signal time ranges and sampling intervals (available only):")
            seen = set()
            for sig in self.x_keys + self.y_keys:
                if sig in seen:
                    continue
                seen.add(sig)
                entry = shot.get(sig, None)
                t = None if not isinstance(entry, dict) else entry.get("time", None)
                if t is None:
                    print(f"  - {sig:30s} ⟶ MISSING")
                    continue
                t = np.asarray(t)
                if t.size < 2:
                    print(f"  - {sig:30s} ⟶ too few samples (N={t.size})")
                    continue
                dt = np.min(np.diff(t))
                print(f"  - {sig:30s} spans {t[0]:.5f} → {t[-1]:.5f}, Δt ≈ {dt:.6f}, N = {len(t)}")

            print(f"\n[INFO] Expected x-window duration: {x_win_sec:.6f} s")
            print(f"[INFO] Expected y-window duration: {y_win_sec:.6f} s")
            print(f"[INFO] Total coverage per window:  {required_span:.6f} s\n")

        results: List[Dict[str, Any]] = []
        t_x_start = start_time
        i = 0

        while True:
            t_x_end = t_x_start + x_win_sec
            t_y_start = t_x_end + self.dt_sec
            t_y_end = t_y_start + y_win_sec

            if t_y_end > end_time:
                break

            strict = self.drop_incomplete

            # X: strict → require all; tolerant → allow per-key missing
            x = self._collect_per_signal_windows(
                shot, self.x_keys, t_x_start, t_x_end,
                allow_missing=(not strict)
            )
            # Y: strict → require all; tolerant → allow per-key missing
            y = self._collect_per_signal_windows(
                shot, self.y_keys, t_y_start, t_y_end,
                allow_missing=(not strict)
            )

            def _present_count(d):
                if d is None:
                    return 0
                return sum(1 for v in d.values() if v is not None)

            n_x_present = _present_count(x)
            n_y_present = _present_count(y)

            if strict:
                # STRICT: all X and all Y must be present
                if (x is not None) and (y is not None) \
                        and (n_x_present == len(self.x_keys)) \
                        and (n_y_present == len(self.y_keys)):
                    results.append({"x": x, "y": y, "window_index": i})
                elif self.verbose:
                    reasons = []
                    if (x is None) or (n_x_present < len(self.x_keys)):
                        reasons.append("X incomplete")
                    if (y is None) or (n_y_present < len(self.y_keys)):
                        reasons.append("Y incomplete")
                    print(f"[seg] shot_window[{i}] skipped ({', '.join(reasons)})")
            else:
                # TOLERANT: keep if at least one X and at least one Y present
                if (n_x_present >= 1) and (n_y_present >= 1):
                    # keep per-key None entries for missing X/Y
                    results.append({"x": x, "y": y, "window_index": i})
                    if self.verbose:
                        def _lens(d):
                            out = {}
                            for k, v in d.items():
                                if v is None:
                                    out[k] = 0
                                else:
                                    tt = v.get("time", None)
                                    out[k] = 0 if tt is None else int(np.asarray(tt).shape[0])
                            return out

                        print(f"[Window {i}] x_len: {_lens(x)}, y_len: {_lens(y)}")
                else:
                    if self.verbose:
                        why = []
                        if n_x_present < 1: why.append("no X present")
                        if n_y_present < 1: why.append("no Y present")
                        print(f"[seg] shot_window[{i}] skipped ({' & '.join(why)})")

            t_x_start += stride
            i += 1

        if self.verbose:
            print(f"[dbg] strict windows:   {len(results) if self.drop_incomplete else 'n/a'}")
            print(f"[dbg] tolerant windows: {len(results) if not self.drop_incomplete else 'n/a'}")

        return results

    # ------------------------------------------------------------------------------------------------------------------
    def _collect_per_signal_windows(
        self,
        shot: Dict[str, Any],
        keys: list[str],
        t_start: float,
        t_end: float,
        allow_missing: bool
    ) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
        """
        Slice each requested key to [t_start, t_end).
        - If `allow_missing=True`: missing/short signals yield `key: None` entry (kept).
        - If `allow_missing=False`: any missing/short signal -> return None (reject window).
        """
        slices: Dict[str, Optional[Dict[str, Any]]] = {}

        for key in keys:
            d = shot.get(key, None)
            times = None if not isinstance(d, dict) else d.get("time", None)
            values = None if not isinstance(d, dict) else d.get("values", None)

            if times is None or values is None:
                if allow_missing:
                    slices[key] = None
                    continue
                else:
                    return None

            t = np.asarray(times)
            v = np.asarray(values)

            # Index within the window
            mask = (t >= t_start) & (t < t_end)
            idx = np.nonzero(mask)[0]

            if len(idx) < self.min_samples:
                if allow_missing:
                    slices[key] = None
                    continue
                else:
                    return None

            lo, hi = int(idx[0]), int(idx[-1]) + 1
            slices[key] = {
                "time": t[lo:hi],
                "values": v[..., lo:hi],
            }

        return slices

