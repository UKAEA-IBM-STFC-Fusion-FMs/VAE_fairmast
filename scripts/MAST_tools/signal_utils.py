# Contributors:
# - Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com).
# Remarks:
# - Based on previous implementation.
# Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html

from typing import Union
import xarray as xr
try:
    from . import store_utils
    from . import constants as cc
except ImportError:
    import store_utils
    import constants as cc


# ======================================================================================================================
class MASTSignalManager:
    """
    Class to process signals from fair MAST bucket
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            manager_id: str = "",
            store_manager_settings: Union[dict, cc.NoneType] = None
    ):
        """
        Attributes
        ----------
        manager_id : str
            User defined manager ID. Default: "".
        store_manager_settings : Union[dict, None]
            Settings for the store manager instance. If None, a generic store manage instance is created with default
            values as defined in the docstrings of store_utils.MASTStorageManager.
            Optional. Default: None.

        Methods
        -------
        get_source_profiles(data_origin, source_name)
            Get source profiles from a given data origin.
        get_signal_values(data_origin, source_name, signal_name)
            Get signal values from a given data origin.
        _set_store_manager(store_manager)
            Set the store_manager instance attribute.

        Remarks
        -------
        - Upon creation of a signal manager instance from the MASTSignalManager class, a store manager (from the
          MASTStorageManager class is created as instance attribute. This facilitates the process to get signal values,
          as no separate store manager instance must be created. However, if an existing store manager instance is
          available, it could be passed

        """

        self.signal_manager_id = manager_id

        if store_manager_settings is None:
            self.store_manager_settings = {}
        else:
            assert isinstance(store_manager_settings, dict), "Type error: invalid store_manager_settings. It must be" \
                                                             " of type dict."
            self.store_manager_settings = store_manager_settings
        self.store_manager = store_utils.MASTStorageManager(**self.store_manager_settings)

    # ------------------------------------------------------------------------------------------------------------------
    def _set_store_manager(self, store_manager):
        """Set the store_manager instance attribute."""
        assert isinstance(store_manager, store_utils.MASTStorageManager), "Type error: invalid store_manager. It must" \
                                                                          " be of type store_utils.MASTStorageManager."
        self.store_manager = store_manager

    # ------------------------------------------------------------------------------------------------------------------
    def get_source_profiles(
           self,
           data_origin: Union[dict, cc.ZarrStoreType],
           source_name: str
    ):
        """
        Get source profiles from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, cc.ZarrStoreType]
            Origin of data for source profile creation.
        source_name : str
            Name of target source.

        Returns
        -------
        xarray.core.dataset.Dataset
            Source profiles from given data origin.

        """
        self.store_manager._check_data_origin(data_origin)  # noqa
        assert isinstance(source_name, str), "Type error: invalid source_name. It must be of type str."

        if isinstance(data_origin, dict):
            # From shot info
            store = self.store_manager.make_shot_store(shot_info=data_origin)
        else:
            # From store
            store = data_origin

        return xr.open_zarr(store=store, group=source_name)

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_values(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Union[str, cc.NoneType] = None,
            verbose: bool = False
    ):
        """
        Get signal values from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : str
            Name of target source.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        [numpy.ndarray, None]
            Signal values, or None if error.

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name,
            data_origin=data_origin,
            source_name=source_name,
            verbose=verbose
        )

        if signal_profile is not None:
            return signal_profile.values
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_times_and_time_type(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Union[str, cc.NoneType] = None,
            verbose: bool = False
    ):
        """
        Get signal values from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : str
            Name of target source.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        [numpy.ndarray, str] or [None, None]
            Signal times with time type, or [None, None] if error.

        """

        signal_profile = self.get_signal_profile(
            signal_name=signal_name,
            data_origin=data_origin,
            source_name=source_name,
            verbose=verbose
        )

        if signal_profile is not None:

            try:
                time_type_ = [str(kk) for kk in signal_profile.coords.keys() if str(kk).startswith("time")][0]
            except IndexError:
                # If here, no time info was found, and so signal is not time-dependent.
                return None, None

            signal_times_ = signal_profile[time_type_].values

            return signal_times_, time_type_
        else:
            # If here, an error occurred while creating the signal profile.
            return None, None

    # ------------------------------------------------------------------------------------------------------------------
    def get_signal_profile(
            self,
            signal_name: str,
            data_origin: Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType],
            source_name: Union[str, cc.NoneType] = None,
            verbose: bool = False
    ):
        """
        Get signal profile from a given data origin.

        Parameters
        ----------
        signal_name : str
            Name of the target signal.
        data_origin : Union[dict, cc.ZarrStoreType, cc.XarrayDatasetType]
            Origin of data for signal value retrieval.
        source_name : str
            Name of target source.
            Optional. Default: None.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        numpy.ndarray or None
            Signal profile, or None if error.

        """
        assert isinstance(signal_name, str), "Type error: invalid source_name. It must be of type str."

        profile = None
        if isinstance(data_origin, cc.XarrayDatasetType):
            # From group profile (i.e., xarray.core.dataset)
            profile = data_origin

        else:
            # From shot info or cc.ZarrStoreType

            try:
                self.store_manager._check_data_origin(data_origin)  # noqa
            except Exception as e:
                if verbose:
                    print(f"Exception: {e}")

            assert isinstance(source_name, str), "Type error: invalid source_name. It must be of type str."

            if isinstance(data_origin, dict):
                # From shot info
                store = self.store_manager.make_shot_store(shot_info=data_origin)
            else:
                # From cc.ZarrStoreType (cc.ZarrFSStoreType or cc.ZarrLocalStoreType)
                store = data_origin

            try:
                profile = xr.open_zarr(store=store, group=source_name)
            except KeyError as e:
                if verbose:
                    print(f"Exception: {e}")

        if profile is not None:
            try:
                return profile[signal_name]
            except KeyError:
                if verbose:
                    print(f"Invalid signal_name {signal_name}.")
                return None
        else:
            # If here, an error occurred while creating the signal profile.
            return None

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_channel_names(store, group, signal_name, verbose=False):
        try:
            profile = xr.open_zarr(store=store, group=group)
            data_array = profile[signal_name]
            non_time_coords = [coord for coord in data_array.coords if coord != "time"]
            if non_time_coords:
                channel_coord = non_time_coords[0]
                return data_array.coords[channel_coord].values
            else:
                return None
        except Exception as e:
            if verbose:
                print(f"Error opening Zarr store: {e}")
            return None


# ----------------------------------------------------------------------------------------------------------------------
def test():

    TESTS_TO_RUN = {
        "signal_values_from_store": True,
        "signal_values_from_shot_info": False,
        "signal_times_from_shot_info": False
    }

    signal_manager = MASTSignalManager()
    # print(type(signal_manager.store_manager))

    # ..................................................................................................................

    shot_info = {"shot_id": 30421, "level": 2, "test_data": False, "local": False, "via_parquet": False}
    source_name = "magnetics"
    signal_name = "flux_loop_flux"

    # ..................................................................................................................
    # Get signal values from store
    if TESTS_TO_RUN["signal_values_from_store"]:

        store_from_shot_info = signal_manager.store_manager.make_shot_store(shot_info=shot_info)
        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name,
            data_origin=store_from_shot_info,
            source_name=source_name,
            verbose=True
        )

        print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal values from shot info

    if TESTS_TO_RUN["signal_values_from_shot_info"]:

        signal_values = signal_manager.get_signal_values(
            signal_name=signal_name,
            data_origin=shot_info,
            source_name=source_name
        )

        print(f"Signal values: {signal_values}\n")

    # ..................................................................................................................
    # Get signal times from shot info

    if TESTS_TO_RUN["signal_times_from_shot_info"]:

        signal_times, signal_type = signal_manager.get_signal_times_and_time_type(
            signal_name=signal_name,
            data_origin=shot_info,
            source_name=source_name
        )

        print(f"Signal type: '{signal_type}'\n")
        print(f"Signal times: {signal_times}\n")


# ======================================================================================================================
if __name__ == "__main__":
    test()
