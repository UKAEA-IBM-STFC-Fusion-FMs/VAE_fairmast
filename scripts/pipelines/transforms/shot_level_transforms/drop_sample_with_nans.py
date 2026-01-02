import numpy as np 


# ======================================================================================================================
class DropSampleWithNans:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, check_inputs: bool = True, check_targets: bool = True, verbose: bool = False):
        """
        - check_inputs / check_targets: which sides to scan for NaNs.
        - Missing signals (values is None) NEVER cause a drop here.
        """
        self.check_inputs = check_inputs
        self.check_targets = check_targets
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def _has_nans(self, entry) -> bool:
        """Return True iff entry has an array with any NaN. Missing entries are OK."""
        if entry is None:
            return False
        # entry is expected to be {"time": <np.ndarray or None>, "values": <np.ndarray or None>}
        vals = entry.get("values", None) if isinstance(entry, dict) else None
        if vals is None:
            return False  # whole signal missing → allowed
        return np.isnan(vals).any()

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, list_windows):
        kept = []
        dropped = 0

        for w in list_windows:
            x_ok = True
            y_ok = True

            if self.check_inputs:
                for _, x_entry in w.get("x", {}).items():
                    if self._has_nans(x_entry):
                        x_ok = False
                        break

            if self.check_targets:
                for _, y_entry in w.get("y", {}).items():
                    if self._has_nans(y_entry):
                        y_ok = False
                        break

            if x_ok and y_ok:
                kept.append(w)
            else:
                dropped += 1

        if self.verbose:
            print(f"[DropSampleWithNans] kept {len(kept)}/{len(list_windows)} (dropped {dropped} with NaNs in present signals)")

        return kept
