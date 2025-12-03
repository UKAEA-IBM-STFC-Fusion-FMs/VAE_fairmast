
import os
import pickle
import joblib
import pandas as pd
import torch
import random
import numpy as np
import itertools
from torch.utils.data import DataLoader

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot

# Compute project root relative to this file
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", "..", ".."
))  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Global reproducibility across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
    Call once at startup, before building datasets/loaders/models.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Needed for strict cuBLAS determinism (matmul). Safe if CUDA isn't present.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN + global deterministic guard
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    except Exception as ee:
        print(f"WARNING - torch exception triggered: {ee}")
        pass

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))


# ----------------------------------------------------------------------------------------------------------------------
def seed_worker(worker_id: int):
    """
    Top-level (picklable) worker init. Derives a per-worker seed from
    PyTorch's worker seed so it's consistent with DataLoader's Generator.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------------------------------------------------
def make_data_generator(seed: int) -> torch.Generator:
    """
    Top-level helper to create a reproducible DataLoader generator.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------------------------------------------------------
def dataloader_seed_parts(seed: int):
    # reuse the top-level function so it's picklable under 'spawn'
    return seed_worker, make_data_generator(seed)


# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(
    max_index = None,
    max_index_for_train = None,
    max_index_for_val = None,
    max_index_for_test = None
    ):
    
    """
    Generate lists of shot IDs for training, testing, and validation.
    These lists can be subsets of the corresponding complete lists.

    Parameters
    ----------
    max_index : int, optional
        If not None, all lists will have the same length given by max_index.
    max_index_for_train : int, optional
        Number of shot IDs for the training set.
        Overrides max_index.
    max_index_for_val : int, optional
        Number of shot IDs for the validation set.
        Overrides max_index.
    max_index_for_test : int, optional
        Number of shot IDs for the testing set.
        Overrides max_index.

    Returns
    -------
    tuple of lists
        Three lists of shot IDs for training, testing, and validation, respectively.

    """

    # Read full data splits
    train_set_full, test_set_full, val_set_full = read_data_split_csv()

    train_set = train_set_full
    test_set = test_set_full
    val_set = val_set_full
    
    # If max_index is provided, override all other limits
    if max_index is not None and max_index > 0:
        train_set = train_set_full[:max_index]
        val_set = val_set_full[:max_index]
        test_set = test_set_full[:max_index]

    # Apply individual limits if provided and positive
    if max_index_for_train is not None and max_index_for_train > 0:
        train_set = train_set_full[:max_index_for_train]

    if max_index_for_val is not None and max_index_for_val > 0:
        val_set = val_set_full[:max_index_for_val]

    if max_index_for_test is not None and max_index_for_test > 0:
        test_set = test_set_full[:max_index_for_test]

    return train_set, test_set, val_set


# ----------------------------------------------------------------------------------------------------------------------
def read_data_split_csv(csv_path="metadata/2025-05-12/data_splits.csv"):
    """Read the csv file containing the lists of shot IDs for
    training, validation and testing.
    """

    full_path = os.path.join(REPO_ROOT, csv_path)
    print(full_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV not found at: {full_path}")

    df = pd.read_csv(full_path)

    shot_ids_for_train = df[df['train'] == True]['shot_id'].tolist()  # noqa
    shot_ids_for_test = df[df['test'] == True]['shot_id'].tolist()  # noqa
    shot_ids_for_val = df[df['val'] == True]['shot_id'].tolist()  # noqa

    return shot_ids_for_train, shot_ids_for_test, shot_ids_for_val


# ----------------------------------------------------------------------------------------------------------------------
def fit_mean_and_std_for_signal_transform(output_sub_dir, train_shots, source_signal_list, verbose=False, local=False):

    if verbose:
        print('\n\n----------TRANSFORM FITTING----------\n')

    preprocessing_train_dataset = MastDataset(
        local=local,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    if verbose:
        print(f"dict_mean_ is {dict_mean_}")
    dict_std_ = get_std_shot(preprocessing_train_dataset)
    if verbose:
        print(f"dict_std_ is {dict_std_}")

    # Save dict_mean and dict_std used!

    output_dir_ = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir_, exist_ok=True)
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir_}")

    with open(output_dir_ + 'dict_mean_shot.pkl', 'wb') as f_:
        pickle.dump(dict_mean_, f_)
    with open(output_dir_ + 'dict_std_shot.pkl', 'wb') as f_:
        pickle.dump(dict_std_, f_)

    return dict_mean_, dict_std_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_datasets(
        sources_and_signals,
        shots,
        sig_tran_map,
        shot_tran,
        local_flag=False,
        return_incomplete_shots=False,
        verbose=False

):
    datasets_ = {"train": None, "val": None, "test": None}

    # ..................................................................................................................
    # Train

    if shots["train"]:
        datasets_["train"] = MastDataset(
            local=local_flag,
            shots_list=shots["train"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # ..................................................................................................................
    # Val

    if shots["val"]:
        datasets_["val"] = MastDataset(
            local=local_flag,
            shots_list=shots["val"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(val_dataset): {len(datasets_['val'])}")

    # ..................................................................................................................
    # Test

    if shots["test"]:
        datasets_["test"] = MastDataset(
            local=local_flag,
            shots_list=shots["test"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
            return_incomplete_shots=return_incomplete_shots,
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    # ..................................................................................................................
    # Return

    return datasets_


# ----------------------------------------------------------------------------------------------------------------------
def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        verbose=False,
        seed: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers=False
):
    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print('\n\n----------DATASET & DATALOADER INITIALIZATION----------\n')

    # ▶ Prepare reproducible seeding parts for DataLoader
    worker_fn = None
    generator = None
    if seed is not None:
        worker_fn = seed_worker
        generator = make_data_generator(seed)

    # sensible default for pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # ..................................................................................................................
    # Train
    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,  # ▶
            generator=generator,  # ▶ controls shuffle order deterministically
            pin_memory=pin_memory,
        )

    # ..................................................................................................................
    # Val
    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,  # ▶ ensures worker RNG is fixed
            generator=generator,  # ▶ reproducible order if shuffle=True
            pin_memory=pin_memory,
        )

    # ..................................................................................................................
    # Test
    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
            worker_init_fn=worker_fn,
            generator=generator,
            pin_memory=pin_memory,
        )

    return dataloaders_


# ======================================================================================================================
class ComposeTransforms(object):
    """Compose transforms and apply them in series checking for None return values

    Parameters
    ----------
    transforms : list[callable[tuple]]
        List containing the names of the transforms
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, transforms):
        self.transforms = transforms

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, sample):
        for transform in self.transforms:
            if sample is None:
                return None
            sample = transform(sample)
        return sample


# ----------------------------------------------------------------------------------------------------------------------
def load_models(data_names, data_dir):
    """Load the PCA and imputer models for the given data names.

    Parameters
    ----------
    data_names : list[str]
        List of data names to load models for.
    data_dir : str
        Root data directory.

    Returns
    -------
    dict
        Dictionary containing the loaded PCA and imputer models.
    """

    pca_models = {}
    imputer_models = {}

    for source_name, signal_name in data_names:
        pca_model_path = f"{data_dir}pca_{signal_name}.joblib"
        imputer_model_path = f"{data_dir}imputer_{signal_name}.joblib"
        pca_models[f"{source_name}-{signal_name}"] = joblib.load(pca_model_path)
        imputer_models[f"{source_name}-{signal_name}"] = joblib.load(imputer_model_path)

    return {"pca": pca_models, "imputer": imputer_models}


def to_dict(obj):
    # Convert attributes of a class instance into a dict
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}