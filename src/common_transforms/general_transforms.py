from collections.abc import Mapping
from typing import Any, Sequence, Optional
import numpy as np
from collections.abc import Mapping
from typing import Any


'''From tokamark/scripts/test_pipeline.py'''
class ModelSpecificTransform:  # TEMPLATE
    """
    Model specific transform.

    Attributes
    ----------
    verbose : bool
        If True, activate verbose mode.

    Methods
    -------
    __call__(shot)
        Call method.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, verbose=False) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        verbose : bool
            If True, activate verbose mode.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method.

        Parameters
        ----------
        shot : Dict[str, Any]
            Target shot.

        Returns
        -------
        dict[str, Any]
            Dictionary with "x" and "y" keys and values from `shot["input"] + shot["actuator"]` and `shot["output"]`
            items, respectively.

        """
        return {
            "x": (
                [data["values"] for var, data in shot["input"].items()]
                + [data["values"] for var, data in shot["actuator"].items()]
            ),
            "y": [data["values"] for var, data in shot["output"].items()],
        }



# ======================================================================================================================
'''Adaptation from tokamark/src/tokamark/tools/transforms/stdscale_transform.py'''
class StdScalingTransform:
    """
    STD scaling transform.

    Methods
    -------
    __call__(dict_)
        It normalizes each sample by subtracting mean and dividing by STD. Outlayers (>3std) are set to NaN.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean: float, std: float) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        mean : float
            Input mean.
        std : float
            Input STD.

        Returns
        -------
        # None  

        """

        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        """
        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Dictionary with "time" and "values" keys with corresponding values.

        Returns
        -------
        dict[str, Any]
            Augmented input dictionary with values normalized per feature.

        """

        values = dict_["values"]

        if values is not None:
            z = (values - self.mean)/self.std
            values[np.abs(z) > 2.705] = np.nan
            values = (values - self.mean) / self.std

        return {"time": dict_["time"], "values": values}

    # ------------------------------------------------------------------------------------------------------------------

class ReplaceNaN():

    def __init__(self,replacement: float):
        self.replacement = replacement
    
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        
        values = dict_["values"]

        if values is not None:
            nan_mask = np.isnan(values)
            dict_["values"][nan_mask] = self.replacement
        
        return dict_
    
class StdDescalingTransform:
    """
    STD descaling transform.

    Methods
    -------
    __call__(dict_)
        Starting from normalized samples multiply by STD and adds mean.
    """
    
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, mean: float, std: float) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        mean : float
            Input mean.
        std : float
            Input STD.

        Returns
        -------
        # None  

        """

        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, dict_: Mapping[str, Any]) -> dict[str, Any]:
        """
        Parameters
        ----------
        dict_ : Mapping[str, Any]
            Dictionary with "time" and "values" keys with corresponding values.

        Returns
        -------
        dict[str, Any]
            Augmented input dictionary with values normalized per feature.

        """

        values = dict_["values"]

        if values is not None:
            values = (values * self.std) + self.mean

        return {"time": dict_["time"], "values": values}

    # ------------------------------------------------------------------------------------------------------------------


class ProbeDiagnosticTransform:  # TEMPLATE
    """
    Model specific transform.

    Attributes
    ----------
    verbose : bool
        If True, activate verbose mode.

    Methods
    -------
    __call__(shot)
        Call method.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, verbose=False) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        verbose : bool
            If True, activate verbose mode.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        """

        self.verbose = verbose

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, shot: Mapping[str, Any]) -> dict[str, Any]:
        """
        Call method.

        Parameters
        ----------
        shot : Dict[str, Any]
            Target shot.

        Returns
        -------
        dict[str, Any]
            Dictionary with "x" and "y" keys and values from `shot["input"] + shot["actuator"]` and `shot["output"]`
            items, respectively.

        """
        return {
            "x": next(iter(shot["input"].values()))["values"],
            "shot_id" : shot['shot_id']
        }