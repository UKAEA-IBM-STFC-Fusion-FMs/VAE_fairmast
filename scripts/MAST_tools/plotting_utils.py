# Contributors:
# - Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com).
# Remarks:
# - Based on previous implementation.
# Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html

import time
import numpy as np
import xarray as xr
from xarray.core.dataarray import DataArray as xarrayDataArray
from xarray.core.dataset import Dataset as xarrayDataset
import matplotlib.pyplot as plt
from typing import Union
import warnings
try:
    from . import store_utils
    from . import signal_utils
    from . import constants as cc
except ImportError:
    import store_utils
    import signal_utils
    import constants as cc


# ======================================================================================================================
class MASTPlottingManager:
    def __init__(
            self,
            manager_id: str = ""
    ):
        """
        Attributes
        ----------
        manager_id : str
            User defined manager ID. Default: "".

        Methods
        -------
        _plot_single_1d_profile(profile, ax)
            Helper function for plotting single 1D profile.
        _check_fig_size(fig_size)
            Helper function for checking fig_size.
        plot_1d_profiles(profiles, fig_size)
            Helper function for plotting 1D profiles, either from xarrayDataArray (single profile) or from xarrayDataset
            (group of profiles).
        plot_signal(data_origin, source_name, signal_name, fig_size)
            Helper function for plotting individual signals.
        plot_group(data_origin, source_name, fig_size)
            Helper function for plotting entire group of signals.
        plot_plasma_current(data_origin, fig_size)
            Helper function for plotting summary__plasma_current signal.
        plot_power_nbi(data_origin, fig_size)
            Helper function for plotting summary__power_nbi signal.
        plot_magnetics(data_origin, fig_size)
            Helper function for plotting magnetics group.
        plot_spectrometer(data_origin, fig_size)
            Helper function for plotting spectrometer_visible group.
        plot_charge_exchange(data_origin, fig_size)
            Helper function for plotting charge_exchange group.
        plot_thomson_scattering(data_origin, fig_size)
            Helper function for plotting thomson_scattering group.
        """

        self.plot_manager_id = manager_id

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _plot_single_1d_profile(
            profile: xr.DataArray,
            ax=None
    ):
        """Helper function for plotting single 1D profile."""

        try:
            profile.plot(ax=ax)
            if ax is None:
                ax = plt.gca()
            ax.grid('on', alpha=0.5)
        except Exception as e:
            print(f"Error: {e}")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_fig_size(
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for checking fig_size."""

        assert isinstance(fig_size, Union[list, set, cc.NoneType]), "Type error: invalid `fig_size` type. It must be" \
                                                                    " list or set."
        if fig_size is not None:
            assert len(fig_size) == 2, "Type error: invalid `fig_size`. It must be a list or set of length 2."

    # ------------------------------------------------------------------------------------------------------------------
    def plot_1d_profiles(
            self,
            profiles: Union[xarrayDataArray, xarrayDataset],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting 1D profiles, either from xarrayDataArray (single profile) or from xarrayDataset
        (group of profiles)."""
        warnings.warn(
            "WARNING: This method misbehaves if not run in a notebook with `%matplotlib notebook` header.")

        assert isinstance(profiles, Union[xarrayDataArray, xarrayDataset]), "Type error: invalid `profiles` argument." \
                                                                            " It must be either xarrayDataArray" \
                                                                            " instance or xarrayDataset instance."
        assert isinstance(fig_size, Union[list, set, cc.NoneType]), "Type error: invalid `fig_size` type. It must be" \
                                                                    " list or set."
        if fig_size is not None:
            assert len(fig_size) == 2, "Type error: invalid `fig_size`. It must be a list or set of length 2."

        if isinstance(profiles, xarrayDataset):
            # Group of profiles from Dataset
            n_rows = int(np.ceil(len(profiles.data_vars) / 2))
            fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2 * n_rows) if fig_size is None else fig_size)
            axes = axes.flatten()

            for ii, name in enumerate(profiles.data_vars.keys()):
                self._plot_single_1d_profile(profiles[name], ax=axes[ii])
        else:
            # Single profile from DataArray
            fig, axes = plt.subplots(1, figsize=(8, 4) if fig_size is None else fig_size)
            self._plot_single_1d_profile(profiles, ax=axes)

        plt.show()
        plt.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_signal(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            source_name: str,
            signal_name: str,
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting individual signals."""

        store = store_utils.MASTStorageManager()._get_store_from_data_origin(data_origin=data_origin)  # noqa.

        self._check_fig_size(fig_size=fig_size)

        profiles = xr.open_zarr(store, group=source_name)
        self.plot_1d_profiles(profiles=profiles[signal_name], fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_group(
            self,
            data_origin: Union[int, cc.ZarrStoreType],
            source_name: str,
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting entire group of signals."""

        warnings.warn(
            "WARNING: This method misbehaves if not run in a notebook with `%matplotlib notebook` header.")

        store = store_utils.MASTStorageManager()._get_store_from_data_origin(data_origin=data_origin)  # noqa.

        self._check_fig_size(fig_size=fig_size)

        try:
            profiles = xr.open_zarr(store, group=source_name)
        except KeyError:
            raise KeyError(f"Group `{source_name}` not found in consolidated metadata.")

        n_rows = int(np.ceil(len(profiles.data_vars) / 2))
        fig, axes = plt.subplots(n_rows, 2, figsize=(8, 2 * n_rows) if fig_size is None else fig_size)
        axes = axes.flatten()

        for ii, name in enumerate(profiles.data_vars.keys()):
            signal = profiles[name]
            if len(signal.shape) == 1:
                self._plot_single_1d_profile(signal, ax=axes[ii])
            elif len(signal.shape) == 2:
                for channel in range(signal.shape[0]):
                    s = signal.isel({signal.dims[0]: channel})
                    s.plot(ax=axes[ii])
            axes[ii].set_title(name)

        for ax in axes:
            ax.grid('on', alpha=0.5)

        plt.show()
        plt.tight_layout()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_plasma_current(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting summary__plasma_current signal."""
        self.plot_signal(data_origin=data_origin, source_name="summary", signal_name="ip", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_power_nbi(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting summary__power_nbi signal."""
        self.plot_signal(data_origin=data_origin, source_name="summary", signal_name="power_nbi", fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_magnetics(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting magnetics group."""
        self.plot_group(data_origin=data_origin, source_name='magnetics', fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_spectrometer(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting spectrometer_visible group."""
        self.plot_group(data_origin=data_origin, source_name='spectrometer_visible', fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_charge_exchange(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting charge_exchange group."""
        if fig_size is None:
            fig_size = [8, 4]

        self.plot_group(data_origin=data_origin, source_name='charge_exchange', fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_thomson_scattering(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            fig_size: Union[list[int], set[int]] = None
    ):
        """Helper function for plotting thomson_scattering group."""
        if fig_size is None:
            fig_size = [8, 6]

        self.plot_group(data_origin=data_origin, source_name='thomson_scattering', fig_size=fig_size)

    # ------------------------------------------------------------------------------------------------------------------
    # def plot_cameras(
    #         self,
    #         data_origin: Union[int, cc.ZarrStoreType],
    #         fig_size: Union[list[int], set[int]] = None
    # ):
    #     # if fig_size is None:
    #     #     fig_size = [8, 4]
    #
    #     self.plot_group(data_origin=data_origin, source_name='cameras', fig_size=fig_size)  # FIXME: No group cameras

    # ------------------------------------------------------------------------------------------------------------------
    # def make_movie(
    #         self,
    #         store
    # ):
    #     raise NotImplementedError  # FIXME: No group cameras

    # output_video_file = "output_video.mp4"
    # fps = 10  # Frames per second
    # frame_width, frame_height = 1000, 500  # Resolution matching figsize in pixels
    #
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
    #
    # try:
    #     profiles = xr.open_zarr(store, group='cameras')
    #
    #     for t in range(1, len(profiles.camera_a.time), 1):
    #         print(t)
    #         if t % 5 != 0:
    #             continue
    #         try:
    #             fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #             profiles['camera_a'].isel(time=t).plot(ax=axes[0])
    #             profiles['camera_b'].isel(time=t).plot(ax=axes[1])
    #
    #             plt.tight_layout()
    #
    #             fig.canvas.draw()
    #             frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    #             # Resize frame to match video resolution
    #             frame = cv2.resize(frame, (frame_width, frame_height))
    #
    #             # Write the frame to the video
    #             video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #
    #             # Close the figure to free memory
    #             plt.close(fig)
    #         except Exception as e:
    #             print(f"Error {e}")
    #
    #         # Release the video writer
    #     video_writer.release()
    #
    # except Exception as e:
    #     print(f"Error: {e}")


# ----------------------------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()

    # ..................................................................................................................
    # Settings

    # Creation of managers
    signal_manager = signal_utils.MASTSignalManager()
    plotting_manager = MASTPlottingManager()

    # Create source from shot_info
    store_from_shot_info = signal_manager.store_manager.make_shot_store(
        shot_info={
            "shot_id": 30421,
            "level": 2,
            "test_data": False,
            "local": False,
            "via_parquet": False
        }
    )
    # print(f"store_from_shot_id.tree() (store store_from_shot_id shot if): {store_from_shot_id.tree()}\n")

    source_profiles = signal_manager.get_source_profiles(
        data_origin=store_from_shot_info,
        source_name="summary"
    )

    # ..................................................................................................................
    # Plot 1d profiles

    plotting_manager.plot_1d_profiles(
        profiles=source_profiles,
        fig_size=[8, 8]
    )

    # ..................................................................................................................
    # Plot specific group

    plotting_manager.plot_plasma_current(data_origin=store_from_shot_info)

    # ..................................................................................................................
    print(f"\n\nElapsed time for execution of main(): {round(time.time() - t0, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    main()
