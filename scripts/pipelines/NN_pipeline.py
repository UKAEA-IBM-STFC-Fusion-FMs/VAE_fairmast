import argparse
import numpy as np
import os
import pandas as pd
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd) + os.sep
sys.path.append(mother_dir)

from MAST_tools.MAST_dataset import MastDataset

from pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from pipelines.transforms.signal_level_transforms.pca_transform import PCATransform 
from pipelines.transforms.shot_level_transforms.time_segmentation_transform import SegmenterTransform
from pipelines.collate_functions.collate_functions import (first_item, TimeWindowSegmentationCollateFn)

from pipelines.configs.config_setup import get_settings

from  pipelines.utils.utils import ( load_models,
                          ComposeTransforms,
                          read_data_split_csv)

#================================================================
        ##########    DATALOADER  ##########
#================================================================
def initialize_pipeline(
    SETTINGS,
    shots_list: list[int],
    data_map_signal_level: dict,
    target_map_signal_level: dict,
    transforms_data_shot_level: callable,
    transforms_target_shot_level: callable  
):
    """Initialize the DataLoader for the MASTDataset.
    
    Parameters
    ----------
    SETTINGS : object
        Configuration settings for the pipeline.
    shots_list : list[int]
        List of shot IDs to load data for.
    data_map_signal_level : dict
        Map of signal-level transforms for data signals.
    target_map_signal_level : dict
        Map of signal-level transforms for target signals.
    transforms_data_shot_level : callable
        Transform to apply at shot level for data signals.
    transforms_target_shot_level : callable
        Transform to apply at shot level for target signals.
    Returns
    -------
    DataLoader
        Initialized DataLoader object.
    """
    
    # Initialize Datasets for training
    data_dataset = MastDataset(
        local=SETTINGS.DATA.local,
        shots_list=shots_list,
        source_signal_list=SETTINGS.DATA.data_names,
        signal_level_transform_map= data_map_signal_level,
        shot_level_transform= transforms_data_shot_level
    )
    
    target_dataset = MastDataset(
        local=SETTINGS.DATA.local,
        shots_list=shots_list,
        source_signal_list=SETTINGS.DATA.target_names,
        signal_level_transform_map=target_map_signal_level,
        shot_level_transform=transforms_target_shot_level
    )
    
    # Initialize DataLoaders for training and validation
    data_loader = DataLoader(
        dataset = data_dataset,
        batch_size=SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers,
        shuffle=True,
        collate_fn=first_item # otherwise pytorch will wrap automatically an item of MastDataset into a list. 
    )
  
    
    target_loader = DataLoader(
        dataset = target_dataset,
        batch_size=SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers,
        shuffle=True,
        collate_fn=first_item # otherwise pytorch will wrap automatically an item of MastDataset into a list. 
    )
    return data_loader, target_loader

#================================================================
    ########## Model Definition and Training ##########
#================================================================
class NeuralNetwork(nn.Module):
    """
    Simple feedforward neural network.

    Architecture
        - Fully connected layer with 64 units
        - ReLU activation
        - Fully connected output layer with 2 units
    """

    def __init__(self, input_size, output_size, l1_size, l2_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, l1_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(l1_size, l2_size)
        self.fc3 = nn.Linear(l2_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
def get_input_output_size(data_loader, target_loader, device='cpu'):
    """
    Get the flattened input/output size from the first batch of data.
    Assumes both loaders are zipped and aligned, and return a list of segments.
    Each segment is a dict with "sources_signals" key.
    """
    
    for x_segments, y_segments in zip(data_loader, target_loader):
        if not x_segments or not y_segments:
            raise ValueError("x_segments or y_segments is empty.")

        # Take the first time window segment
        x_segment = x_segments[0]
        y_segment = y_segments[0]

        x_signals = x_segment["sources_signals"]
        y_signals = y_segment["sources_signals"]

        if not x_signals or not y_signals:
            raise ValueError("First segment in x_batch or y_batch is missing 'sources_signals'.")

        try:
            x_tensor_list = [signal_dict["values"].to(device).flatten() for signal_dict in x_signals]
            y_tensor_list = [signal_dict["values"].to(device).flatten() for signal_dict in y_signals]

            x_flat = torch.cat(x_tensor_list, dim=0)
            y_flat = torch.cat(y_tensor_list, dim=0)
        except Exception as e:
            raise ValueError(f"Failed to concatenate tensors: {e}")

        input_dim = x_flat.shape[0]
        output_dim = y_flat.shape[0]

        print(f"NN size: input_dim = {input_dim}, output_dim = {output_dim}")
        return input_dim, output_dim

    raise RuntimeError("No data found in loaders.")
  
def compare_last_first_tensor_shapes(x_list,y_list):
    """ Check that the shape of the last element in the lists is coinsistent 
    with the number of signals expected, e.g., the nr_signals found in the first element

    Parameters
    ----------
    x_list : list[Tensor]
        Each entry in y_list is a list of tensors each one of shape :
        (nr_features, time length).
        Each one corresponding to a segment (or time-window) of the original input signals.
    y_list : list[Tensor]
        Each entry in y_list is a list of tensors each one of shape :
        (nr_features, time length)..
        Each one corresponding to a segment (or time-window) of the original target signals.

    Returns
    -------
    x_list, y_list minus the last tensors 
    if these have different shapes from the first tensor
    """

    if not x_list or not y_list:
        print("Warning: One or both lists are empty, returning as-is.")
        return x_list, y_list
    if len(x_list) <= 1 or len(y_list) <= 1:
        return x_list, y_list

    # Compare number of tensors (nr_signals) in first and last entries
    input_num_signals_0 = len(x_list[0])  # Number of tensors in first x_list entry
    output_num_signals_0 = len(y_list[0])  # Number of tensors in first y_list entry
    input_num_signals_1 = len(x_list[-1])  # Number of tensors in last x_list entry
    output_num_signals_1 = len(y_list[-1])  # Number of tensors in last y_list entry

    if input_num_signals_0 != input_num_signals_1 or output_num_signals_0 != output_num_signals_1:
        print(f"Signal count mismatch: first x/y num_signals ({input_num_signals_0}, {output_num_signals_0}), "
              f"last x/y num_signals ({input_num_signals_1}, {output_num_signals_1}). Removing last entries.")
        x_list = x_list[:-1]
        y_list = y_list[:-1]
    
    return x_list, y_list
    
# Manual batch splitting inside the training loop.
def rebatch_dict_of_lists(batch, batch_size, min_final_batch_size=10):
    x_list = batch['x']
    y_list = batch['y']
    # Each entry in x_ y_list is a list of tensors each one of shape :
    # (nr_features, time-window length).
    
    ## HINT:
    # The last tensor might have a smaller nr_signals. In fact, during the segmentation
    # in time windows, some signals might drop the last segment if the time length 
    # is not long enough. 

    # Check that the shape of the last element in the lists is coinsistent 
    # with the number of signals expected, e.g., the nr_signals found in the first batch
    x_list, y_list = compare_last_first_tensor_shapes(x_list, y_list)
        
    total = len(x_list)
    batches = []

    if total < batch_size:
        return [{'x': x_list, 'y': y_list}]
    
    i = 0
    while i < total and i + batch_size <= total:
        batches.append({
            'x': x_list[i:i+batch_size],
            'y': y_list[i:i+batch_size]
        })
        i += batch_size
    
    remaining = total - i
    if remaining > 0:
        if remaining < min_final_batch_size:
            last_batch = batches[-1]
            last_batch['x'].extend(x_list[i:])
            last_batch['y'].extend(y_list[i:])
        else:
            batches.append({
                'x': x_list[i:],
                'y': y_list[i:]
            })
            
    return batches

def get_loss_per_signal(output, y_batch, y_tensor, log_file=None):
    y_signals_shape = [signal.shape for signal in y_batch[0]]
    signal_sizes = [torch.tensor(shape).prod().item() for shape in y_signals_shape]
    
    # Store per-signal MSEs across the batch
    per_signal_diffs = [[] for _ in signal_sizes]

    for i in range(output.shape[0]):  # output.shape[0] is the number of batches
        out_i = output[i]
        tgt_i = y_tensor[i]
        
        # Split based on signal sizes
        out_signals = torch.split(out_i, signal_sizes)
        tgt_signals = torch.split(tgt_i, signal_sizes)
        
        for j, (o_sig, t_sig) in enumerate(zip(out_signals, tgt_signals)):
            mse = F.mse_loss(o_sig, t_sig, reduction='mean').item()
            per_signal_diffs[j].append(mse)

    # Now per_signal_diffs[j] is a list of MSEs for signal j across the batch
    # You can print or log average per-signal error:
    avg_signal_errors = [sum(diffs) / len(diffs) for diffs in per_signal_diffs]
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"Avg MSE per signal: {['%.2e' % e for e in avg_signal_errors]}\n")
    else:
        print(f"Avg MSE per signal: {['%.2e' % e for e in avg_signal_errors]}")
    
    
def run_model(
    model,
    device, 
    data_loader,
    target_loader,
    batch_size,
    min_batch_size,
    num_epochs, 
    current_process="training",
    criterion=nn.MSELoss(),
    optimiser=None,
    log_file=None,
    verbose = True):
    """
    Train the NeuralNetwork model.

    Parameters
    ----------
    model : NeuralNetwork
        The neural network model to train.
    device : torch.device
        The device to train the model on.
    data_loader_train, : DataLoader
        DataLoader for the training data.
    target_loader_train, : DataLoader
        DataLoader for the training target.
    batch_size : int
        Number of batches in the training data.
    min_batch_size : int, optional
        Minimum size of the final batch, by default 10.
    num_epochs : int, optional
        Number of epochs to train the model.
    criterion : nn.Module, optional
        Loss function to use, by default nn.MSELoss().
    optimiser : torch.optim.Optimizer, optional
        Optimiser to use for training, by default None. If None, Adam optimiser is used
    """
    collator = TimeWindowSegmentationCollateFn()

    loss_vs_epoch = []
    for epoch in range(num_epochs):
        
        verbose and print(f"{current_process} epoch: {epoch}")
        
        if current_process == "training":
            model.train()
        else:
            model.eval()
            torch.set_grad_enabled(False)
            
        # Loop through base batches from the DataLoader
        running_loss = 0.0
        for data_list, target_list in zip(data_loader, target_loader):
            base_batch = collator(data_list, target_list)
            
            if batch_size is not None and min_batch_size is not None:
                batches = rebatch_dict_of_lists(base_batch, batch_size, min_batch_size)
            else:
                batches = [base_batch]
            
            for batch in batches:
                x_batch = batch['x']
                y_batch = batch['y']

                x_tensor_list = []
                y_tensor_list = []
                    
                for x_signals, y_signals in zip(x_batch, y_batch):
                    x_flat = torch.cat([sig.flatten() for sig in x_signals], dim=0)
                    y_flat = torch.cat([sig.flatten() for sig in y_signals], dim=0)
                    x_tensor_list.append(x_flat)
                    y_tensor_list.append(y_flat)

                # Stack into batch tensors
                x_tensor = torch.stack(x_tensor_list)  # [batch_size, input_dim]
                y_tensor = torch.stack(y_tensor_list)  # [batch_size, target_dim]
                
                x_tensor = x_tensor.to(device)
                y_tensor = y_tensor.to(device)
                
                verbose and print(f"x shape {x.shape()}, y shape: {y.shape()}")
                
                output = model(x_tensor)
                loss = criterion(output, y_tensor)
                
                if current_process == "testing":
                    get_loss_per_signal(output, y_batch, y_tensor, log_file)
                    
                if current_process == "training":
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                running_loss += loss.item()
                    
        verbose and print(f"Epoch: {epoch}, {current_process} loss: {running_loss/len(data_loader):.2e}")
        loss_vs_epoch.append(running_loss/len(data_loader))    
        
    return loss_vs_epoch


#================================================================
    ########## END of Model Definition and Training ##########
#================================================================

if __name__== "__main__":
    
    # Determine device to train on
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
        
    #============== INPUT and SET-UP SECTION ==============#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_file",
        type=str,
        default="config_pipeline_my.json",
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    # Load configuration from JSON file
    config_file_path =  args.pipeline_config_file

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")       
    
    # Initialize SETTINGS object
    SETTINGS = get_settings(config_file_path)
    
    all_source_signal_list = SETTINGS.DATA.data_names + SETTINGS.DATA.target_names
    # Load models from joblib files
    model_dictionary = load_models(all_source_signal_list, 
                                   SETTINGS.LOCAL_PATHS.joblib_directory)
    
    #============== END INPUT and SET-UP SECTION ==============#

    
    
    ########## INITIALIZATION SECTION ##########
    train_shots, test_shots, val_shots = read_data_split_csv(SETTINGS.LOCAL_PATHS.data_split_csv_path)
    train_shots = train_shots[:SETTINGS.TRAINING.num_train_samples]  
    val_shots = val_shots[:SETTINGS.TRAINING.num_val_samples] # For testing purposes, limit the number of validation shots


    # Define transform pipelines for data and target signals
    
    # 1- Maps of transforms at signal level
    data_map_signal_level= {var: ComposeTransforms([
        ImputerTransform(model_dictionary["imputer"][var], 
                         SETTINGS.LOCAL_PATHS.average_values_file_path),
        PCATransform(model_dictionary["pca"][var])
    ])
        for var in [f'{source}-{signal}' for source, signal in SETTINGS.DATA.data_names]
    }
    
    target_map_signal_level = {var: ComposeTransforms([
        ImputerTransform(model_dictionary["imputer"][var], 
                         SETTINGS.LOCAL_PATHS.average_values_file_path),
        PCATransform(model_dictionary["pca"][var])
    ])
        for var in [f'{source}-{signal}' for source, signal in SETTINGS.DATA.target_names]
    }
    
    transforms_data_shot_level = ComposeTransforms([
                SegmenterTransform(
                    SETTINGS.TIME_SEGMENTATION.time_window_sec, 
                    SETTINGS.TIME_SEGMENTATION.stride_sec,
                    SETTINGS.TIME_SEGMENTATION.offset)
            ]
        )
                                                    
    transforms_target_shot_level = ComposeTransforms([
                SegmenterTransform(
                    SETTINGS.TIME_SEGMENTATION.offset,
                    SETTINGS.TIME_SEGMENTATION.stride_sec,
                    offset=0.0)
            ]
        )
    
    data_loader_train, target_loader_train = initialize_pipeline(
        SETTINGS,
        train_shots,
        data_map_signal_level,
        target_map_signal_level,
        transforms_data_shot_level,
        transforms_target_shot_level
    )
    
    data_loader_validation, target_loader_validation = initialize_pipeline(
        SETTINGS,
        val_shots,
        data_map_signal_level,
        target_map_signal_level,
        transforms_data_shot_level,
        transforms_target_shot_level
    )
    ########## END INITIALIZATION SECTION ##########
     
     
    ########## MODEL DEFINITION ##########
    input_size, output_size = 0, 0
    # Get input and output size from the first batch of data
    try:
        input_size, output_size =  get_input_output_size(
            data_loader_train, 
            target_loader_train,
            device=device)
        print(f"Input size: {input_size}, Output size: {output_size}")
    except ValueError as e:
        print(f"Error getting input/output size: {e}")
        sys.exit(1)


    model = NeuralNetwork(
        input_size=input_size, 
        output_size = output_size,
        l1_size=SETTINGS.NEURALNET.l1_size,
        l2_size=SETTINGS.NEURALNET.l2_size
        ).to(device)
    
    criterion = nn.MSELoss() # Loss function to use, by default MSELoss.
    optimiser = torch.optim.Adam(model.parameters(), lr=SETTINGS.NEURALNET.lr)
    ########## END MODEL DEFINITION ########## 
    
    ########## MODEL TRAINING & VALIDATION ##########
    output_filename=SETTINGS.LOCAL_PATHS.data_output_directory + "NeuralNetwork.txt"
    current_process = "training"
    train_loss = run_model(
        model,
        device,
        data_loader_train,
        target_loader_train,
        SETTINGS.TRAINING.train_batch_size,
        SETTINGS.TRAINING.min_batch_size,
        SETTINGS.TRAINING.num_epochs,
        current_process,
        criterion,
        optimiser,
        log_file=None,
        verbose = True
        )
    
    current_process = "validating"
    val_loss = run_model(
        model,
        device,
        data_loader_validation,
        target_loader_validation,
        SETTINGS.TRAINING.train_batch_size,
        SETTINGS.TRAINING.min_batch_size,
        SETTINGS.TRAINING.num_epochs,
        current_process,
        criterion,
        optimiser,
        log_file=None
        )
    
    # current_process = "testing"
    # eval_loss = run_model(
    #     model,
    #     device,
    #     data_loader_validation,
    #     target_loader_validation,
    #     SETTINGS.TRAINING.train_batch_size,
    #     SETTINGS.TRAINING.min_batch_size,
    #     SETTINGS.TRAINING.num_epochs,
    #     current_process,
    #     criterion,
    #     optimiser,
    #     log_file=None
    #     )
    

    print("Training completed successfully.")
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'SETTINGS': SETTINGS.config,
        'model_hyperparameters': {
        'input_size': input_size,
        'output_size': output_size,
        'l1_size': SETTINGS.NEURALNET.l1_size,
        'l2_size': SETTINGS.NEURALNET.l2_size,
        'learning_rate': SETTINGS.NEURALNET.lr
        },
        'training_hyperparameters': {
            'num_epochs': SETTINGS.TRAINING.num_epochs,
            'batch_size': SETTINGS.TRAINING.train_batch_size,
            'device': str(device)
        },
        'metrics': {
            'train_loss': train_loss,
            'eval_loss': val_loss
        }}, f"model{SETTINGS.NEURALNET.lr}.pth")

