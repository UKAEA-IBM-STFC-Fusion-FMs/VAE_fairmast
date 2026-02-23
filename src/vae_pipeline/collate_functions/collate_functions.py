
"""This file contains the collate functions for batching samples in a DataLoader. 
    
    Expected output of the collate function is a dictionary with two keys: 
        return {'x': batched_x, 'y': batched_y}
    
    see transforms.SegmenterTransform for more details on the input data format.
"""      
from collections import defaultdict, Counter
import torch
from typing import List, Dict, Any

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