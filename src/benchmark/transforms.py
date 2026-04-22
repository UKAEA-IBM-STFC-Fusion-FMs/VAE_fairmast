from collections.abc import Mapping
from typing import Any, Sequence, Optional

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

