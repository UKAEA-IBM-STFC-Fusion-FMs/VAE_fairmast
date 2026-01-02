import numpy as np

# ======================================================================================================================
class TruncationTransform:
    """
    Shot-level transform that truncates all signals in a shot to the minimum end time
    across all available time-series. This ensures consistent signal duration when
    signals have different time extents or sampling frequencies. Robust to missing signals and metadata.

    Usage
    -----
    transform = TruncationTransform()
    truncated_shot = transform(shot)

    Input
    -----
    shot : dict
        Dictionary of signals:
            {
                'signal_name': {
                    'time': np.ndarray,     # shape (T,)
                    'values': np.ndarray    # shape (C, T)
                },
                ...,
                'shot_id': str or int      # optional metadata
            }

    Output
    ------
    truncated_shot : dict
        Same structure as input, but all signals truncated to share the same end time.

    Notes
    -----
    - Only signal entries (dicts with 'time' and 'values') are truncated.
    - Signals with no valid data before the truncation point will raise an error.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot):
        # --- 1) Collect valid end-times from signal entries only ---
        end_times = []
        for key, data in shot.items():
            if not isinstance(data, dict):
                continue
            t = data.get("time", None)
            v = data.get("values", None)
            if t is None or v is None:
                continue
            t = np.asarray(t)
            if t.size == 0:
                continue
            # ignore non-finite times
            t = t[np.isfinite(t)]
            if t.size == 0:
                continue
            end_times.append(float(np.max(t)))

        # If nothing valid, return shot unchanged (nothing to truncate)
        if not end_times:
            return dict(shot)

        min_end_time = float(min(end_times))

        # --- 2) Truncate each signal to <= min_end_time; pass non-signal entries through ---
        new_shot = {}
        for key, data in shot.items():
            if not isinstance(data, dict):
                # metadata or other non-signal payload: keep as-is
                new_shot[key] = data
                continue

            t = data.get("time", None)
            v = data.get("values", None)
            if t is None or v is None:
                # keep explicit "missing" convention
                new_shot[key] = {"time": None, "values": None}
                continue

            t = np.asarray(t)
            v = np.asarray(v)

            # Guard empty arrays
            if t.size == 0 or v.size == 0:
                new_shot[key] = {"time": None, "values": None}
                continue

            # keep finite times and truncate to min_end_time
            finite_mask = np.isfinite(t)
            mask = finite_mask & (t <= min_end_time)
            idx = np.nonzero(mask)[0]

            if idx.size == 0:
                # no valid samples before truncation point → mark missing
                new_shot[key] = {"time": None, "values": None}
            else:
                lo, hi = int(idx[0]), int(idx[-1]) + 1
                new_shot[key] = {
                    "time": t[lo:hi],
                    "values": v[..., lo:hi],
                }

        return new_shot

    # ------------------------------------------------------------------------------------------------------------------
