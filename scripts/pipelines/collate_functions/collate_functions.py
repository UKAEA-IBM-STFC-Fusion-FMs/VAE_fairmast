
"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      
from collections import defaultdict, Counter
import torch

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


def conv1d_vae_collate_fn_old(batch, verbose = False):
    """_summary_

    Parameters
    ----------
    batch : list of samples

    sample = {
        "signal1": [Tensor(C1, T1.0), Tensor(C1, T1.1), ...],  # one per time window
        "signal2": [Tensor(C2, T2.0), Tensor(C2, T2.1), ...],
        ...
    }
    
    For each signal, there is a list of tensors one for each temporal window. 

    Returns
    -------
    defaultdict
    {
        "signal1" : [
            torch.stack(for tensors T in windows 0),
            torch.stack(for tensors T in windows 1),
            ...
            ],
        "signal2" : [
            torch.stack(for tensors T in windows 0),
            torch.stack(for tensors T in windows 1),
            ...
            ]
    }
    
    EXAMPLE: 
    batch = [
        {
            "S1": [torch.tensor([1]), torch.tensor([2])],
            "S2": [torch.tensor([10]), torch.tensor([20])]
        },
        {
            "S1": [torch.tensor([3])],
            "S2": [torch.tensor([30])]
        }
    ]
    
    {
        "S1": [
            torch.stack([torch.tensor([1]), torch.tensor([3])]),  # index 0
            torch.stack([torch.tensor([2])])                      # index 1 
        ],
        "S2": [
            torch.stack([torch.tensor([10]), torch.tensor([30])]),
            torch.stack([torch.tensor([20])])
        ]
    }
 
    """
    collated = defaultdict(lambda : defaultdict(list))
    
    for sample in batch:
        if isinstance(sample, list):
            continue
        for signal_name, list_of_tensors in sample.items():
            for nr, tensor in enumerate(list_of_tensors):
                collated[signal_name][nr].append(tensor)
    
    
    # stack into a single tensor per signal
    final_batch = {}
    for signal_key, index_dict in collated.items():
        final_batch[signal_key] = []
        # Sort indices to maintain order
        for i in sorted(index_dict.keys()):
            
            # Some tensors might have wrong shape
            tensors = index_dict[i]
            
            # Some tensor might have an extra entry due to the temporal window segmentation
            tensor_ref_size = min(t.shape[-1] for t in tensors)
            
            # Crop tensors along the last dimension if necessary
            cropped = [t[..., :tensor_ref_size] for t in tensors]
                
            final_batch[signal_key].append(torch.stack(cropped))

    return final_batch
         
def nested_defaultdict():
    return defaultdict(list)
       
class Conv1dVAECollate():
    """_summary_

    Parameters
    ----------
    batch : list of samples

    sample = {
        "signal1": [Tensor(nr_features1, time-length1), Tensor(nr_features1, time-length1), ...],  # one per time window
        "signal2": [Tensor(nr_features2, time-length2), Tensor(nr_features2, time-length2), ...],
        ...
    }
    
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
                for tensor in list_of_tensors:
                    tensors.append(tensor)
                    
                    if len(tensors) == self.targeted_number_tensors:
                        collated[signal_name][index[signal_name]] = torch.stack(tensors) 
                        index[signal_name] += 1
                        tensors = []
                        
                # Collate remaining tensors
                if tensors:
                    collated[signal_name][index[signal_name]] = torch.stack(tensors)

        # Check if any signal has no groups formed
        empty_signals = [s for s, groups in collated.items() if len(groups) == 0]
        if empty_signals:
            if self.verbose:
                print(f"Warning: No groups formed for signals: {empty_signals}. Consider lowering targeted_number_tensors.")
            raise ValueError(f"Collate failed: not enough tensors to form a single group for signals: {empty_signals}")

        return collated
