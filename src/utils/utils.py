
import os
import pandas as pd
import random
import numpy as np

# Compute project root relative to this file
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", "..", ".."
))  

from MAST_tools.MAST_dataset import MastDataset, CachedDataset
from torch.utils.data import DataLoader

def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False,
        cache_data=True,
        return_incomplete_shots = False,
        store_mast_settings={}
    ):
    
    datasets_ = {"train": None, "val": None, "test": []}
    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if shots[data_set_type]:
            datasets_[data_set_type] = MastDataset(
                local=local_flag,
                shots_list=shots[data_set_type],
                source_signal_list=sources_and_signals,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transforms,
                return_incomplete_shots = return_incomplete_shots,
                remove_outliers = True,
                store_manager_settings = store_mast_settings
            )
            
    if cache_data:
        datasets_["train"] = CachedDataset(datasets_["train"])
        datasets_["val"]   = CachedDataset(datasets_["val"]) 
        datasets_["test"] = CachedDataset(datasets_["test"]) 
           
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers = False
    ):
    
    dataloaders_ = {"train": None, "val": None, "test": None}

    data_set_types = ["train", "val", "test"]
    
    for data_set_type in data_set_types:
        if datasets[data_set_type]:
            dataloaders_[data_set_type] = DataLoader(
                dataset=datasets[data_set_type],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_function,
                persistent_workers = persistent_workers
            )

    return dataloaders_

# ----------------------------------------------------------------------------------------------------------------------
def get_train_test_val_shots(
    max_index = None,
    max_index_for_train = None,
    max_index_for_val = None,
    max_index_for_test = None,
    shuffle = False,
    seed = None,
    csv_path =  "tokamark/src/tokamark/metadata/TokaMark_data_splits.csv"
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
    shuffle: bool
        True if we need shuffled samples.
    seed: int 
        For reproducibility of the rnd sequence.

    Returns
    -------
    tuple of lists
        Three lists of shot IDs for training, testing, and validation, respectively.

    """

    # Read full data splits
    train_set_full, test_set_full, val_set_full = read_data_split_csv(csv_path)

    if shuffle:
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError(f"Seed must be an integer, got {type(seed).__name__}")
            random.seed(seed)  
            
        random.shuffle(train_set_full)
        random.shuffle(test_set_full)
        random.shuffle(val_set_full)
        
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
def read_data_split_csv(csv_path):
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


