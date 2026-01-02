# Contributors:
# - Rodrigo Ordonez-Hurtado (rodrigo.ordonez.hurtado@ibm.com).
# Remarks:
# - Based on previous implementation, but augmented to support zarr 3.x.x.
# Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html

# TODO: Evaluate if level, test_data, and local should be class attributes instead.

import zarr
import zarr.storage
import fsspec
import s3fs
import pandas as pd
import warnings
from typing import Union
import logging
from urllib.error import HTTPError
from pprint import pprint
import time
try:
    from . import constants as cc
except ImportError:
    import constants as cc

logging.getLogger('asyncio').setLevel(logging.CRITICAL)


# ======================================================================================================================
class MASTStorageManager:
    def __init__(
            self,
            base_fsspec_protocol: str = "simplecache",
            target_fsspec_protocol: str = "s3",
            s3_endpoint_url: str = "https://s3.echo.stfc.ac.uk",
            base_remote_parquet_path: Union[str, cc.NoneType] = "https://mastapp.site/parquet",
            base_local_zarr_path: Union[str, cc.NoneType] = "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp",
            base_local_parquet_path: Union[str, cc.NoneType] = None,
            store_manager_id: str = ""
    ):
        """
        Attributes
        ----------
        base_fsspec_protocol: str
            Base protocol used by fsspec. Some supported protocols include:
            "blockcache":
                - With this option, data is downloaded block-wise.
                - Restrictions:
                    - It has a storage/OS combination which supports sparse files.
                    - The backend implementation uses files which derive from AbstractBufferedFile.
                    - The library you pass the resultant object to accepts generic python file-like objects.
            "filecache":
                - Works for all file system implementations, and provides a real local file for other libraries to use.
            "simplecache":
                - Same as “filecache”, except without options for cache expiry and to check original source.
                - Only option guaranteed to be thread/process-safe.
            Full list of supported protocols obtained via fsspec.available_protocols().
            More info available at https://filesystem-spec.readthedocs.io/en/latest/features.html.
            Default: "simplecache".

        target_fsspec_protocol : str
            Target filesystem protocol for the selected base fsspec protocol.
            Default: "s3".

        s3_endpoint_url : str
            Endpoint of the cloud S3 bucket used for remote data pulling (i.e., for local=False).
            Default: "https://s3.echo.stfc.ac.uk".

        base_remote_parquet_path : Union[str, NoneType]
            Local root path used for local data pulling (i.e., for local=False) in parquet format.
            Default: "https://mastapp.site/parquet/".

            base_local_zarr_path : Union[str, NoneType]
            Local root path used for local data pulling (i.e., for local=True) in zarr format.
            Default: "/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp", corresponding to the CSD3 cluster.

        base_local_parquet_path : Union[str, NoneType]
            Local root path used for local data pulling (i.e., for local=True) in parquet format.
            Default: None.

        store_manager_id : str
            User defined manager ID. Default: "".

        Methods
        -------
        _is_digit(item)
            Check if provided item is of type digit.
        _get_store_from_data_origin(data_origin)
            Auxiliary function to get Zarr store instance from a given data origin.
        _parse_shot_info_dict(shot_info)
            Parse dictionary with shot information.
        _check_level(level)
            Check MAST data level.
        _check_data_origin(data_origin)
            Check MAST data origin.
        _check_shot_id(shot_id)
            Check shot ID.
        _check_list_of_shot_ids(shot_ids)
            Check list of shot IDs.
        _create_fs_remote(library, warn)
            Create filesystem instance either using fsspec or s3fs for remote data request.
        _read_parquet_data(path)
            Read MAST data using parquet pipeline.
        _read_fsspec_listdir(path, local)
            Invoke list dir method of fsspec filesystem instance.
        list_all_shots(level, test_data, local, via_parquet)
            Get a list of available MAST shot IDs.
        list_shots_by_signal_availability(availability_data_filepath, required_signals)
            List shot IDs following composite condition for signal availability and given availability file.
        get_all_sources(shot_ids, level, test_data, local, via_parquet)
            Return a dictionary with all available sources per shot ID.
        get_all_signals(shot_ids, level, test_data, local, via_parquet)
            Return a dictionary with all available signals per shot ID.
        make_shot_store(shot_id, verbose)
            Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.
        make_shot_group(data_origin, verbose)
            Make a shot group from data origin (either a Zarr store or shot info).

        Pending methods
        ---------------

        print_store(...)
            ...
        is_signal_in_store(...)
            ...
        print_signals_in_group(...)
            ...

        """

        self.base_fsspec_protocol = base_fsspec_protocol
        self.target_fsspec_protocol = target_fsspec_protocol
        self.s3_endpoint_url = s3_endpoint_url
        self.base_local_zarr_path = base_local_zarr_path
        self.base_local_parquet_path = base_local_parquet_path
        self.base_remote_parquet_path = base_remote_parquet_path

        self.fs_local_fsspec = fsspec.filesystem("file")
        self.fs_remote_fsspec = self._create_fs_remote(library="fsspec", warn=False)
        self.fs_remote_s3fs = self._create_fs_remote(library="s3fs")

        self.store_manager_id = store_manager_id

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _is_digit(item):
        """Check if provided item is of type digit."""
        try:
            return item.isdigit()
        except AttributeError:
            return item.is_integer()

    # ------------------------------------------------------------------------------------------------------------------
    def _get_store_from_data_origin(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
    ):
        """
        Auxiliary function to get Zarr store instance from a given data origin.

        Parameters
        ----------
        data_origin : Union[dict, cc.ZarrStoreType]
            Origin of data for group creation. It can be a dictionary with shot information (as in the class method
            self.make_shot_store()) or a Zarr store.

        Returns
        -------
        zarr.storage.FsspecStore
            Instance of Zarr store.

        """
        self._check_data_origin(data_origin=data_origin)
        if isinstance(data_origin, dict):
            # data_origin is a dict with shot info
            return self.make_shot_store(shot_info=data_origin)
        else:
            # data_origin is a Zarr store
            return data_origin

    # ------------------------------------------------------------------------------------------------------------------
    def _parse_shot_info_dict(
            self,
            shot_info: dict,
    ):
        """
        Parse dictionary with shot information.

        Parameters
        ----------
        shot_info : dict
            Dictionary with shot information required for store creation. All dictionary keys and types are as follows:
                'shot_id' : int
                    ID of target shot to be pulled from MAST database.
                'level' : int
                    Target level for the MAST data to be pulled.
                    Default: 2.
                'test_data' : bool
                    If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not
                    available for locally stored data (i.e, if local=True).
                    Default: False.
                'local' : bool
                    If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise
                    it is pulled from the registered remote data repository (e.g., a cloud S3 bucket).
                    Default: False.
                'via_parquet' : bool, optional
                    If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline
                    (e.g., fsspec.filesystem) is used.

        Returns
        -------
        list
            List of parsed dictionary items.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.

        """

        assert isinstance(shot_info, dict), "Type error: invalid shot_info. It must be dict."

        shot_id = shot_info.get("shot_id")
        self._check_shot_id(shot_id=shot_id)

        level = shot_info.get("level", 2)
        self._check_level(level=level)

        local = shot_info.get("local", False)
        assert isinstance(local, bool), "Type error: invalid argument local. It must be of type bool."

        test_data = shot_info.get("test_data", False)
        assert isinstance(test_data, bool), "Type error: invalid argument test_data. It must be of type bool."

        via_parquet = shot_info.get("via_parquet", False)
        assert isinstance(via_parquet, bool), "Type error: invalid argument via_parquet. It must be of type bool."

        return shot_id, level, local, test_data, via_parquet

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_level(
            level: int,
    ):
        """Check MAST data level."""
        assert isinstance(level, int), "Type error: invalid argument level. It must be of type int."
        assert level in [1, 2], "Value error: level must be in [1, 2]."

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_data_origin(
            data_origin: Union[dict, cc.ZarrStoreType],
    ):
        """Check MAST data origin."""
        assert isinstance(data_origin, cc.DataSourceType), "Type error: invalid data_origin. It must be dict or store."

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_shot_id(
            shot_id: int,
    ):
        """Check shot ID."""
        assert isinstance(shot_id, int), f"Type error: invalid shot_id {shot_id}. It must be of type int."

    # ------------------------------------------------------------------------------------------------------------------
    def _check_list_of_shot_ids(
            self,
            shot_ids: list[int],
    ):
        """Check list of shot IDs."""
        assert isinstance(shot_ids, list), f"Type error: invalid list of shot IDs. It must be a list of int entries."
        for id_ in shot_ids:
            self._check_shot_id(id_)

    # ------------------------------------------------------------------------------------------------------------------
    def _create_fs_remote(
            self,
            library: str,
            warn: bool = True
    ):
        """
        Create filesystem instance either using fsspec or s3fs for remote data request.

        Parameters
        ----------
        library : str
            The target library for filesystem instance creation.
        warn : bool
            Boolean flag to warn about issues during filesystem instance creation.

        Returns
        -------
        filesystem instance
            A filesystem instance, either fsspec.filesystem or s3fs.S3FileSystem.

        Raises
        ------
        NotImplementedError
            If a library other than fsspec or s3fs is provided.

        """

        if library == "fsspec":
            if warn:
                warnings.warn("WARNING: Library `fsspec` cannot create asynchronous instances of file systems.")

            return fsspec.filesystem(
                protocol=self.base_fsspec_protocol,
                target_protocol=self.target_fsspec_protocol,
                target_options=dict(anon=True, endpoint_url=self.s3_endpoint_url),
                # cache_storage='.cache',  # Uncomment and define cache folder, if required.
                # asynchronous=True  # REMARK: This is allowed, but it does not make the instance asynchronous.
            )

        elif library == "s3fs":
            return s3fs.S3FileSystem(
                anon=True,
                endpoint_url=self.s3_endpoint_url,
                asynchronous=True
            )
        else:
            raise NotImplementedError(f"Error: Library {library} not supported.")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _read_parquet_data(
            path: str
    ):
        """Read MAST data using parquet pipeline."""
        try:
            return pd.read_parquet(path=path)
        except HTTPError:
            raise FileNotFoundError(f"No data available for path {path}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"No data available for path {path}.")

    # ------------------------------------------------------------------------------------------------------------------
    def _read_fsspec_listdir(
            self,
            path: str,
            local: bool = False
    ):
        """Invoke list dir method of fsspec filesystem instance."""
        try:
            if local:
                return self.fs_local_fsspec.ls(path)
            else:
                return self.fs_remote_fsspec.ls(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"No data available for path {path}.")

    # ------------------------------------------------------------------------------------------------------------------
    def list_all_shots(
            self,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False
    ):
        """
        Get a list of available MAST shot IDs.

        Parameters
        ----------
        level : int
            Target level for the MAST data to be pulled. Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Default: False.
        local : bool, optional
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Default: False.
        via_parquet : bool, optional
            If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline (e.g.,
            fsspec.filesystem) is used.

        Returns
        -------
        list
            List of all available sources in the dataset.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.
        FileNotFoundError
            If local or remote path not found.

        """

        self._check_level(level=level)
        test_and_level_case = f"{'test/' if test_data else ''}level{level}"

        if via_parquet:
            # Parquet pipeline
            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path
            full_path = f"{base_parquet_path}/{test_and_level_case}/shots"
            summary = self._read_parquet_data(path=full_path)
            raw_shot_ids = summary["shot_id"].values
        else:
            # FSSpec pipeline
            if local:
                local_path = f"{self.base_local_zarr_path}/{test_and_level_case}/"
                all_filenames = self._read_fsspec_listdir(local_path, local=True)
            else:
                remote_path = f"/mast/{test_and_level_case}/shots/"
                all_filenames = [item["Key"] for item in self._read_fsspec_listdir(remote_path, local=False)]

            raw_shot_ids = [
                filename.split("/")[-1].split(".zarr")[0] for filename in all_filenames if filename.endswith(".zarr")
            ]

        shot_ids = [
            int(raw_shot_id) for raw_shot_id in raw_shot_ids if self._is_digit(raw_shot_id)
        ]
        shot_ids.sort()

        return shot_ids

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def list_shots_by_signal_availability(
            availability_data_filepath: str,
            required_signals: dict
    ):
        """
        List shot IDs following composite condition for signal availability and given availability file.

        Parameters
        ----------
        availability_data_filepath : str
            Path to suitable csv file with signal availability.
        required_signals: dict
            Dictionary with required signal availability.
            Example: {"thomson_scattering": ["n_e"], "summary": ["power_nbi", "ip"]}

        Returns
        -------
        list
            List of shot IDs.

        """

        availability_data = pd.read_csv(availability_data_filepath)

        source_signals_to_have = []
        for kk, vv in required_signals.items():
            source_signals_to_have += ([f"{kk}__{val}" for val in vv])

        # REMARK: Suggested "is True" for line below causes error due to misbehaviour, thus noqa in place.
        composite_condition = availability_data[source_signals_to_have[0]] == True  # noqa

        for signal in source_signals_to_have[1:]:
            # REMARK: Suggested "is True" for line below causes error due to misbehaviour, thus noqa in place.
            composite_condition = composite_condition & (availability_data[signal] == True)  # noqa

        return list(availability_data.loc[composite_condition]["shot_id"])

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_sources(
            self,
            shot_ids: Union[cc.NoneType, list[int]] = None,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False
    ):
        """
        Return a dictionary with all available sources per shot ID.

        Parameters
        ----------
        shot_ids : Union[None, list[int]]
            Target shot IDs to be checked in the MAST database. If None is provided, all available shots are checked.
            Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Optional. Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Optional. Default: False.
        local : bool, optional
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Optional. Default: False.
        via_parquet : bool, optional
            If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline (e.g.,
            fsspec.filesystem) is used.
            Optional. Default: False.

        Returns
        -------
        dict
            Dictionary with available sources per shot ID.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.

        """

        self._check_level(level=level)
        if shot_ids is not None:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        test_and_level_case = f"{'test/' if test_data else ''}level{level}"

        if via_parquet:
            # Parquet pipeline

            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path
            full_path = f"{base_parquet_path}/{test_and_level_case}/sources"
            df = self._read_parquet_data(path=full_path)
            if shot_ids:
                composite_condition = df["shot_id"] == shot_ids[0]
                for shot_id in shot_ids[1:]:
                    composite_condition = composite_condition | (df["shot_id"] == shot_id)
                df = df.loc[composite_condition]

            source_info = {}
            all_shot_ids = list(set(df["shot_id"]))
            for id_ in all_shot_ids:
                sub_df = df.loc[df["shot_id"] == id_]
                source_info[id_] = list(sub_df["name"])

            return source_info

        else:
            # FSSpec pipeline

            if shot_ids is None:
                shot_ids = self.list_all_shots(level=level, test_data=test_data, local=local, via_parquet=False)

            source_info = {}
            for id_ in shot_ids:
                if local:
                    local_path = f"{self.base_local_zarr_path}/{test_and_level_case}/{id_}.zarr"
                    group = zarr.open_group(store=local_path, mode="r")
                else:
                    remote_shot_path = f"/mast/{test_and_level_case}/shots/{id_}.zarr"
                    group = zarr.open_group(
                        store=f"{self.target_fsspec_protocol}:/{remote_shot_path}",
                        mode="r",
                        storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                    )

                source_info[id_] = list(group.keys())

            return source_info

    # ------------------------------------------------------------------------------------------------------------------
    def get_all_signals(
            self,
            shot_ids: Union[cc.NoneType, list[int]] = None,
            level: int = 2,
            test_data: bool = False,
            local: bool = False,
            via_parquet: bool = False,
            verbose: bool = False
    ):
        """
        Return a dictionary with all available signals per shot ID.

        Parameters
        ----------
        shot_ids : Union[None, list of ints]
            Target shot IDs to be checked in the MAST database. If None is provided, all shots are checked.
            Default: None.
        level : int
            Target level for the MAST data to be pulled.
            Default: 2.
        test_data : bool
            If True, the target shot is pulled from test data, otherwise it is pulled from curated data.
            Default: False.
        local : bool, optional
            If True, it checks locally stored data (e.g., in the CSD3 cluster), otherwise it looks into the registered
            remote data repository (e.g., a cloud S3 bucket).
            Default: False.
        via_parquet : bool, optional
            If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline (e.g.,
            fsspec.filesystem) is used.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        dict
            Dictionary with available signals per shot ID.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.

        Remarks
        -------
            - Time signals (e.g., time, time_bes, time_mirnov, time_omaha, time_saddle) are not retrieved as signals for
              via_parquet = True.

        """

        self._check_level(level=level)
        if shot_ids is None:
            shot_ids = self.list_all_shots(level=level, test_data=test_data, local=local, via_parquet=via_parquet)
        else:
            self._check_list_of_shot_ids(shot_ids=shot_ids)

        test_and_level_case = f"{'test/' if test_data else ''}level{level}"

        if via_parquet:
            # Parquet pipeline

            base_parquet_path = self.base_local_parquet_path if local else self.base_remote_parquet_path

            signal_info = {}
            for ii, shot_id_ii in enumerate(shot_ids):
                full_path = f"{base_parquet_path}/{test_and_level_case}/signals?shot_id={shot_id_ii}"
                df = self._read_parquet_data(path=full_path)
                signal_info[shot_id_ii] = [f"{row['source']}__{row['name']}" for index, row in df.iterrows()]
                if verbose:
                    print(f"Processed shots: {ii} out of {len(shot_ids)}.")
            return signal_info

        else:
            # FSSpec pipeline

            signal_info = {}
            for shot_id in shot_ids:
                if local:
                    local_path = f"{self.base_local_zarr_path}/{test_and_level_case}/{shot_id}.zarr"
                    group = zarr.open_group(store=local_path, mode="r")
                else:
                    remote_shot_path = f"/mast/{test_and_level_case}/shots/{shot_id}.zarr"
                    group = zarr.open_group(
                        store=f"{self.target_fsspec_protocol}:/{remote_shot_path}",
                        mode="r",
                        storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                    )

                group_members = group.members(1)
                for item in group_members:
                    member = None
                    try:
                        group, member = item[0].split("/")
                        signal_info[shot_id].append(f"{group}__{member}")
                    except KeyError:
                        signal_info[shot_id] = [f"{group}__{member}"]
                    except ValueError:
                        pass
                        # print(f"Skipped item: {item[0]}")

            return signal_info

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_store(
            self,
            shot_info: dict,
            verbose: bool = False
    ):
        """
        Make a Zarr store (either LocalStore or FsspecStore) for a given target shot.

        Parameters
        ----------
        shot_info : dict
            Dictionary with shot information required for store creation. All dictionary keys and types are as follows:
                'shot_id' : int
                    ID of target shot to be pulled from MAST database.
                'level' : int
                    Target level for the MAST data to be pulled.
                    Default: 2.
                'test_data' : bool
                    If True, the target shot is pulled from test data, otherwise it is pulled from curated data. Not
                    available for locally stored data (i.e, if local=True).
                    Default: False.
                'local' : bool
                    If True, the target shot is pulled from locally stored data (e.g., in the CSD3 cluster), otherwise
                    it is pulled from the registered remote data repository (e.g., a cloud S3 bucket).
                    Default: False.
                'via_parquet' : bool, optional
                    If True, it retrieves information via pandas.read_parquet pipeline. Otherwise, alternative pipeline
                    (e.g., fsspec.filesystem) is used.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        Zarr store
            Either LocalStore or FsspecStore instance.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.
        NotImplementedError
            For not implemented pipelines.

        """

        shot_info = self._parse_shot_info_dict(shot_info=shot_info)
        shot_id, level, local, test_data, via_parquet = shot_info
        test_and_level_case = f"{'test/' if test_data else ''}level{level}"

        if via_parquet:
            # Parquet pipeline
            raise NotImplementedError("Parquet pipeline not yet implemented.")
            # TODO: Try implementing parquet pipeline to avoid the NotImplementedError risen above.

        else:
            if local:
                zarr_filepath = f"{self.base_local_zarr_path}/{test_and_level_case}/{shot_id}.zarr"
                store = zarr.storage.LocalStore(root=zarr_filepath)
            else:
                zarr_filepath = f"/mast/{test_and_level_case}/shots/{shot_id}.zarr"
                store = zarr.storage.FsspecStore(
                    fs=self.fs_remote_s3fs,
                    read_only=True,
                    path=zarr_filepath
                )

            if verbose:
                print(f"{'LocalStore' if local else 'FsspecStore'} store for shot {shot_id} created.")

            return store

    # ------------------------------------------------------------------------------------------------------------------
    def make_shot_group(
            self,
            data_origin: Union[dict, cc.ZarrStoreType],
            verbose: bool = False
    ):
        """
        Make a shot group from data origin (either a Zarr store or shot info).

        Parameters
        ----------
        data_origin : Union[dict, ZarrStoreType]
            Origin of data for group creation. It can be a dictionary with shot information (as in the class method
            self.make_shot_store()) or a Zarr store.
        verbose : bool
            If True, verbose mode is activated.
            Default: False.

        Returns
        -------
        zarr.core.group.Group
            Zarr group created from data origin.

        Raises
        ------
        AssertionError
            If an assertion error is triggered.
        NotImplementedError
            For not implemented pipelines.

        """

        self._check_data_origin(data_origin=data_origin)

        if isinstance(data_origin, cc.ZarrStoreType):
            # Create group from store
            store = data_origin
            group = zarr.open_group(store=store, mode='r')
            if verbose:
                print(f"Group for store with path {store.path} created.")
        else:
            # Create group from shot_id

            shot_info = self._parse_shot_info_dict(shot_info=data_origin)
            shot_id, level, local, test_data, via_parquet = shot_info
            test_and_level_case = f"{'test/' if test_data else ''}level{level}"

            if via_parquet:
                # Parquet pipeline
                raise NotImplementedError("Parquet pipeline not yet implemented.")
                # TODO: Implement parquet pipeline to avoid the NotImplementedError risen above.
            else:
                # Non-parquet pipeline
                if local:
                    # Create group by implicitly creating a writable LocalStore
                    local_path = f"{self.base_local_zarr_path}/{test_and_level_case}/{shot_id}.zarr"
                    group = zarr.open_group(store=local_path, mode="r")
                    # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation
                else:
                    # Create group by implicitly creating a read-only FsspecStore
                    remote_shot_path = f"/mast/{test_and_level_case}/shots/{shot_id}.zarr"
                    group = zarr.open_group(
                        store=f"{self.target_fsspec_protocol}:/{remote_shot_path}",
                        mode="r",
                        storage_options={"anon": True, "endpoint_url": self.s3_endpoint_url}
                    )
                    # Source: https://zarr.readthedocs.io/en/latest/user-guide/storage.html#implicit-store-creation

                if verbose:
                    print(f"Group for shot {shot_id} created.")

        if verbose:
            print("Group tree:\n")
            print(group.tree())

        return group


# ----------------------------------------------------------------------------------------------------------------------
def main():
    TESTS_TO_RUN = {  # noqa
        "get_all_shot_ids": True,
        "get_all_sources": True,
        "get_all_signals": True,
        "make_group_from_store": True,
        "make_group_from_shot_info": True,
        "check_signal_availability": True
    }
    t0 = time.time()

    store_manager = MASTStorageManager()

    # ..................................................................................................................
    # List all shot IDs for the entire dataset, using different pipelines

    if TESTS_TO_RUN["get_all_shot_ids"]:

        t0_parquet = time.time()
        all_shots_ids_parquet = set(
            store_manager.list_all_shots(level=2, test_data=False, local=False, via_parquet=False)
        )
        print(f"Elapsed time (parquet pipeline): {round(time.time() - t0_parquet, 2)} s")
        print(f"Shot IDs (parquet): {list(all_shots_ids_parquet)}\n")
        print(f"Number of parquet shots: {len(all_shots_ids_parquet)}\n")

        t0_fsspec = time.time()
        all_shots_ids_ffspec = set(
            store_manager.list_all_shots(level=2, test_data=False, local=False, via_parquet=False)
        )
        print(f"Elapsed time (fsspec pipeline): {round(time.time() - t0_fsspec, 2)} s")
        print(f"Shot IDs (ffspec) : {list(all_shots_ids_ffspec)}\n")
        print(f"Number of fsspec shots: {len(all_shots_ids_ffspec)}\n")

        print(f"Number of parquet-only shots: {len(all_shots_ids_parquet - all_shots_ids_ffspec)}\n")
        parquet_only_shots = list(all_shots_ids_parquet - all_shots_ids_ffspec)
        parquet_only_shots.sort()
        print(f"parquet-only shots: {parquet_only_shots}\n")

        print(f"Number of fsspec-only shots: {len(all_shots_ids_ffspec - all_shots_ids_parquet)}\n")
        fsspec_only_shots = list(all_shots_ids_ffspec - all_shots_ids_parquet)
        fsspec_only_shots.sort()
        print(f"fsspec-only shots: {fsspec_only_shots}\n")

        print(f"Number of common shots: {len(all_shots_ids_ffspec.intersection(all_shots_ids_parquet))}\n")

    # ..................................................................................................................
    # List all sources

    if TESTS_TO_RUN["get_all_sources"]:

        all_sources = store_manager.get_all_sources(
            shot_ids=[30471],  # Use None for entire dataset.
            level=2,
            test_data=False,
            local=False,
            via_parquet=False,
        )
        print(f"all_sources: {all_sources}\n")

    # ..................................................................................................................
    # List all signals

    if TESTS_TO_RUN["get_all_signals"]:

        all_signals = store_manager.get_all_signals(
            shot_ids=[30471],  # Use None for entire dataset.
            level=2,
            test_data=False,
            local=False,
            via_parquet=False,
            verbose=True
        )
        print(f"all_signals:")
        pprint(all_signals)

    # ..................................................................................................................
    # Make group for a given shot

    # Via existing store object
    if TESTS_TO_RUN["make_group_from_store"]:

        store_ = store_manager.make_shot_store(
            shot_info={"shot_id": 30471, "level": 2, "test_data": False, "local": False, "via_parquet": False}
        )
        group_from_store = store_manager.make_shot_group(data_origin=store_)
        print(f"group_from_store.tree() (group from store): {group_from_store.tree()}\n")

    # Directly from shot_info
    if TESTS_TO_RUN["make_group_from_shot_info"]:

        group_from_shot_id = store_manager.make_shot_group(
            data_origin={"shot_id": 30471, "level": 2, "test_data": False, "local": False, "via_parquet": False}
        )
        print(f"group_from_shot_id.tree() (group from shot ID): {group_from_shot_id.tree()}\n")

    # ..................................................................................................................
    # List all shots IDs for given signal availability

    if TESTS_TO_RUN["check_signal_availability"]:

        signal_availability_file = "../../metadata/2025-04-17/data_level2_signal_availability.csv"
        dict_target_signals = {
            "thomson_scattering": ["n_e"],
            "spectrometer_visible": ["filter_spectrometer_bes_voltage"],
            "summary": ["power_nbi", "ip"],
        }

        print("\nSignals to be checked for simultaneous availability across all shots:")
        pprint(dict_target_signals)

        filtered_ids = store_manager.list_shots_by_signal_availability(
            availability_data_filepath=signal_availability_file,
            required_signals=dict_target_signals
        )

        print(f"\nfiltered_ids ({len(filtered_ids)} shots):")
        pprint(filtered_ids)

    # ..................................................................................................................

    print(f"\n\nElapsed time for execution of main(): {round(time.time() - t0, 2)} s")


# ======================================================================================================================
if __name__ == "__main__":
    main()
