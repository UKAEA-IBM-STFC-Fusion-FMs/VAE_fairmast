
"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      
from collections import defaultdict, Counter
import torch
from typing import List, Dict, Any

def first_item(batch):
    return batch[0]
    
class TimeWindowSegmentationCollateFn:
    """
    Customized collate function to collate samples obtained from SegmenterTransform.
    
     Parameters
    ----------
    list_x(y): list(dict)
        A list of sub-dictionaries each one containing a time segment 
        of the original dictionary, for all the signals in the original shot. 
        
        This is the return value of segment_shot in SegmenterTransform.          
    """
    
    def __call__(self, list_x, list_y):
        all_x_segments = []
        all_y_segments = []
        
        len_x, len_y = len(list_x), len(list_y)
        
        if len_x is None or len_y is None:
            print("Warning: problem with data lengths after segmentation: list_x or list_y is None")
            return None

        if len_y == 0 or len_x == 0:
            print("Warning: problem with data lengths after segmentation")
            print(f"Lengths of lists: len(list_y) = {len(list_y)}, len(list_x) = {len(list_x)}")
            return None
        
        # If the lengths of the lists are not equal, we truncate them removing 
        if len_x != len_y:
            if len_x > len_y:
                list_x = list_x[len_x - len_y:]
            elif len_y > len_x:
                list_y = list_y[len_y - len_x:]
            
        for x_segment, y_segment in zip(list_x, list_y):
            # Extract x values
            x_values = [
                signal_dict["values"]
                for signal_dict in x_segment["sources_signals"]
            ]
            # shape: [num_signals, nr_features, time_window_length]
            
            # Extract y values
            y_values = [
                signal_dict["values"]
                for signal_dict in y_segment["sources_signals"]
            ]
            # shape: [num_signals, nr_features, time_window_length]
            
            all_x_segments.append(x_values) # shape: [list_x_length, num_signals, nr_features, time_window_length]
            all_y_segments.append(y_values) # shape: [list_y_length, num_signals, nr_features, time_window_length]
        
        if not all_x_segments or not all_y_segments:
            return None  # or return empty batch dicts
        
        return {'x': all_x_segments, 'y': all_y_segments}


def beta_vae_collate_fn(batch):
    """Custom collate function for β-VAE training"""
    print(f"Collating β-VAE batch of size {len(batch)}")

    # Flatten the batch of lists into a single list
    flattened_batch = [item for sublist in batch for item in sublist]
    print(
        f"Number of signal segments from batch = {len(batch)} shots is N = {len(flattened_batch)}"
    )

    # Group by signal name
    signal_groups = defaultdict(list)
    for item in flattened_batch:
        signal_groups[item["signal_name"]].append(item["data"])

    # Convert to tensors for each signal group
    batched_signals = {}
    for signal_name, data_list in signal_groups.items():
        try:
            batched_signals[signal_name] = torch.stack(
                [torch.from_numpy(data) for data in data_list]
            )
        except Exception as e:
            print(f"Error batching signal {signal_name}: {e}")
            continue

    return batched_signals
         
def nested_defaultdict():
    return defaultdict(list)
       
class Conv1dVAECollate_multiple_signals():
    """DEPRECATED, NOT MANTAINED.

    Parameters
    ----------
    batch : list of samples

    sample = {
        "signal1": [Tensor(nr_features1, time-length1), Tensor(nr_features1, time-length1), ...],  # one per time window
        "signal2": [Tensor(nr_features2, time-length2), Tensor(nr_features2, time-length2), ...],
        ...
    }
    Each sample derives from one item of the dataset, i.e., it represents one particular shot_id.
    All signals in the same sample have the same number of tensors, although tensors shape may differ.
    For each signal, there is a list of tensors one for each temporal window. 

    Returns
    -------
    defaultdict
    
    collated = {
        "signal1": {
            0: Tensor(targeted_number_tensors, nr_features1, time-length1),
            1: Tensor(targeted_number_tensors, nr_features1, time-length1),
            ...
        },
        "signal2": {
            0: Tensor(targeted_number_tensors, nr_features2, time-length2),
            1: Tensor(targeted_number_tensors, nr_features2, time-length2),
            ...
        },
        ...
    }

    
    If the temporal window does not contain the same number of tensors as 
    given by targeted_number_tensors, tensors from the next time window are
    appended.
    """
    
    def __init__(self, signals, targeted_number_tensors, verbose = False):
        self.targeted_number_tensors = targeted_number_tensors
        self.verbose = verbose
        self.signals = signals

    def __call__(self, batch):
        collated = defaultdict(nested_defaultdict)
        
        index = {}
        for signal_name in self.signals:
            index[signal_name] = 0

        for sample in batch:
            if isinstance(sample, list):
                continue
        
            for signal_name, list_of_tensors in sample.items():
                tensors = []
                for ii in range(0,len(list_of_tensors),self.targeted_number_tensors):
                    tensors = list_of_tensors[ii:ii+self.targeted_number_tensors]
                    collated[signal_name][index[signal_name]] = torch.stack(tensors)
                    index[signal_name] += 1

        return collated


class WindowsCollate:

    def __call__(self, batch: List[Dict[str, Any]]):
        all_windows = []
        lengths = []

        for sample in batch:
            # Skip empties and normalize [sample] → sample
            if not sample:
                lengths.append(0)
                continue

            if isinstance(sample, list):
                sample = sample[0]
                if not sample:
                    lengths.append(0)
                    continue

            if not isinstance(sample, dict) or len(sample) == 0:
                lengths.append(0)
                continue

            # windows: List[Tensor(C, T)]
            windows = next(iter(sample.values()), [])
            if not windows:
                lengths.append(0)
                continue

            
            cleaned = []
            for j, w in enumerate(windows):
                if not isinstance(w, torch.Tensor): 
                    continue
                if not torch.isfinite(w).all(): 
                    print("Tensor window contains contain non-finite entries.")
                    continue
                cleaned.append(w)

            t = torch.stack(cleaned, dim=0)   # (len(sample), C, T)
            all_windows.append(t)
            lengths.append(t.shape[0])

        if not all_windows:
            # Return a consistent empty batch; we don't know (C, T) here, so (0, 0, 0) is safest.
            # If you know C,T at construction time, pre-store them and return (0, C, T)
            empty_x = torch.empty(0, 0, 0)  # or torch.empty(0, C, T)
            return {"x": empty_x, "lengths": torch.zeros(0, dtype=torch.int32)}
     
        x = torch.cat(all_windows, dim=0)  # (N_total, C, T)
        return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.int32)}
       

class Conv1dVAECollate():
    """Collate samples in a batch.

    Parameters
    ----------
    batch : list of samples

    sample = {
        "signal_name": [Tensor(nr_features1, time-length1), Tensor(nr_features1, time-length1), ...],  # one per time window
    }
    Each sample derives from one item of the dataset, i.e., it represents one particular shot_id.

    Returns
    -------
    defaultdict
    
    collated = {
        "signal_name": {
            0: Tensor(targeted_number_tensors, nr_features1, time-length1),
            1: Tensor(targeted_number_tensors, nr_features1, time-length1),
            ...
        }
    }

    """
    
    def __init__(self, targeted_number_tensors, verbose = False):
        self.targeted_number_tensors = targeted_number_tensors
        self.verbose = verbose

    def __call__(self, batch):
        collated = defaultdict(list)
        index = 0
        
        for sample in batch:
            
            if len(sample) == 0:
                continue

            if isinstance(sample, list):
                sample = sample[0]
            try:
                sample.items()
            except:
                continue
            
            for signal, list_of_tensors in sample.items():
                tensors = []

                for i in range(0,len(list_of_tensors),self.targeted_number_tensors):
                    tensors = list_of_tensors[i:i+self.targeted_number_tensors]
                    collated[index] = torch.stack(tensors)
                    index += 1
                # In conv1d_vae we only use one signal at a time. 
                # If more than one signal is provided, the loop breaks after first iteration
                break  

        return collated
                

def test_Conv1dVAECollate_multiple_signals():
    # Arrange
    signals = ["signal1", "signal2"]
    targeted_number_tensors = 2
    collate_fn = Conv1dVAECollate_multiple_signals(signals, targeted_number_tensors)

    # Create dummy batch
    batch = [
        {
            "signal1": [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)],
            "signal2": [torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2)]
        },
        {
            "signal1": [torch.randn(2, 2), torch.randn(2, 2)],
            "signal2": [torch.randn(1, 2), torch.randn(1, 2)]
        }
    ]

    collated = collate_fn(batch)

def test_Conv1dVAECollate():
    # Arrange
    signals = ["signal1"]
    targeted_number_tensors = 2
    collate_fn = Conv1dVAECollate(signals, targeted_number_tensors)

    # Create dummy batch
    batch = [
        {
            "signal1": [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)],
        },
        {
            "signal1": [torch.randn(2, 2), torch.randn(2, 2)],
        }
    ]
    print(f"batch: {batch}")
    collated = collate_fn(batch)
    print(f"collated {collated}")
    
if __name__ == "__main__":
    test_Conv1dVAECollate()