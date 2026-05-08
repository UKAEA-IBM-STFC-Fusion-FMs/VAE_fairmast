import argparse
import os
import torch
import sys
from typing import List

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from src.benchmark.configs.benchmark_setup import get_benchmark_settings
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE

def load_benchmark_settings(config_file_path: str):
    """Load JSON settings."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"JSON configuration file not found: {config_file_path}")
    try:
        return get_benchmark_settings(config_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON config from '{config_file_path}': {e}") from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load task YAML and pipeline JSON configuration."
    )
    parser.add_argument(
        "--config_task_file_path",
        default="",
        type=str,
        help="Path to configuration YAML task file."
    )
    parser.add_argument(
        "--config_benchmark_file_path",
        default="src/benchmark/configs/task1_1_config.json",
        type=str,
        help="Path to configuration JSON file for the pipeline."
    )
    return parser.parse_args()


def load_vae_model(config_path:str):
    """
    config_path : str
        Path to the config.json containing parameters for initializing the model
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.") 
    else:
        try:
            settings = get_settings(config_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
            return None
    
    try:
        model = beta_VAE(settings)
    except Exception as e:
        print(f"Error in initializing vae model: {e}")
        return None
            
    return model
    

def create_vae_dictionary(vae_dictionary, type_of_signal, list_of_signals, SETTINGS, model_sub_paths):
    """ Loads VAE models into a dictionary for signals in the list.

       Reinforce the one-to-one correspondence between input or actuator signals and VAE models.

    Parameters
    ----------
    vae_dictionary : _type_
       vae_dictionary = {"input": {}, "actuator": {}, "output": {}}
    type_of_signal : str
        either "input" or "actuator" or "output"
    list_of_signals : list[str]
        List of signals from task config file
    SETTINGS : SettingsBenchmark
        Settings from json file
    model_sub_paths : list[str]
        paths to models

    Return
    --------
        Filled vae_dictionary
    """
    if not list_of_signals:
        return vae_dictionary

    if type_of_signal not in [ "input","actuator", "output"]:
        raise ValueError(f"Type of signal specified {type_of_signal} not in the list of correct keys:  [input, actuator,output] ")

    for i, (source, signal_name) in enumerate(list_of_signals):
        key = f"{source}-{signal_name}"
        vae_dictionary[type_of_signal][key] = None

        for model_path in model_sub_paths:
            if signal_name in model_path:
                
                full_path = os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, model_path)

                # store model keyed by source-signal
                vae_dictionary[type_of_signal][key] = load_vae_model(full_path)
                break

    num_inputs = len(list_of_signals)
    num_loaded = len(vae_dictionary[type_of_signal])

    if num_inputs != num_loaded:
        if type_of_signal in ["input", "actuator"]:
            raise ValueError(
                f"The number of input signals ({num_inputs}) does not match the number of VAE "
                f"models loaded ({num_loaded})."
            )

    return vae_dictionary
    

def process_data(
    models: List[beta_VAE], 
    batched_data: List[torch.Tensor],
    sentinel_value):
    """
    - If model is not None
        For each pair of batched_data and model we get the latent space representation of the 
        data by calling the model encoder. 

    - If model is None
        For batched_data without a paired model, data are reshaped to ndim = 2.

    Assumes:
    1-  Batch_data and models are correctly alligned. 
        For instance, the first entry in models is associated with the first entry in batched_data. 

    2-  The number of models and entries in batched_data must be the same. 

    Parameters
    ----------
    models :  list[beta_VAE]
        List of pre-trained beta_VAE models or None entries.
    batched_data : list[torch.Tensor]
        List of batched tensors in the real space, each tensor has at least 2 dimensions.
    sentinel_value : same type as data in batch
        This is a placeholder for NaN entries. Its value is 99, outside data distribution.

    Returns
    -------
    data : list[torch.Tensor]
        List of batched tensors after processing. 
        The processing consits in two mutually exclusive options:

            1) The tensors in the batch are compressed in the latent space representation, if
            corresponding VAEs exist in models.

            2) The tensors in the batch are kept in their representations in the real space 
            but reshaped to agree with the dimesnion of compressed tensors, i.,e., 2D

    masks : list[torch.Tensor]
        - If model is not None:
        mask[b, :] = 1 iff sample b had nr of sentinels smaller than a certain cut-off else zeros.
        mask is 1 for all latent features of good samples, 0 otherwise.
        
        - If model is not None:
        mask[b, j] = 1 iff that specific flattened entry is not sentinel.
        mask is 1 per-entry in real space representation.

    encode_masks : list[torch.Tensor]
        1 if signal was encoded and passed cut-off.
        0 if signal did not pass cut-off and was set to zero.
        0 if model does not exist.
    """

    # Sanity check
    if len(models) != len(batched_data):
        raise ValueError(
            f"Allignment problem: \
            the length of the list of models: ({len(models)}), is different \
            from the number of data given in the list of data: ({len(batched_data)})"
        )

    # Return values
    data: List[torch.Tensor] = []   # One process data (entry) for each signal in batch_data 
    masks: List[torch.Tensor] = [] # One mask (entry) for each signal in batch_data 
    encode_masks :  List[torch.Tensor] = []  # One mask (entry) for each signal in batch_data 

    # Encode or reshape tensors
    for model, batch in zip(models, batched_data):
        
        # Check nr. of dimensions
        if batch.ndim < 2:
            raise ValueError("Each batch must have at least 2 dimensions [B, ...]")

        # Batch size
        B = batch.shape[0]

        # Filter invalid samples in the batch based on a cut-off of 0.05 of invalid entries per sample
        dims = tuple(range(1, batch.ndim))
        invalid_entries = torch.isclose(batch, sentinel_value)

        # Initialize mask for samples to encode
        encode_mask = torch.zeros(B, device=batch.device, dtype = batch.dtype)

        # Check data for encoding
        if model is not None:
        
            p = next(model.parameters())
            if batch.device != p.device or batch.dtype != p.dtype:
                raise ValueError(f"Batch is not on the same device as the model: {p.device}")

            if sentinel_value.device != batch.device or sentinel_value.dtype != batch.dtype:
                raise ValueError(f"The 'sentinel_value' is not on the same device or same type as the sample\
                                   device: {sentinel_value.device}, {batch.device},\
                                   types:  {sentinel_value.dtype} {batch.dtype}")

            # Initialize to zero
            z = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            mask = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            
            invalid_fraction_per_sample = invalid_entries.float().mean(dim=dims)  # [B]
            tau = 0.05 
            valid_samples = invalid_fraction_per_sample <= tau  # bool [B]

            if valid_samples.any():
                mask[valid_samples, :]  = 1
                with torch.no_grad():
                    # Clone to avoid corruption of original data
                    x = batch.clone()
                    # Ground invalid entries with zero
                    x[invalid_entries] = 0
                    # Encode valid samples
                    z[valid_samples] = model.encode(x[valid_samples])[0]
                    # Update encode mask
                    encode_mask[valid_samples] = 1
                
        else:
            z = batch.reshape(B, -1).clone()
            mask = (~torch.isclose(z, sentinel_value)).to(dtype=batch.dtype)
            z = z * mask

        # Ensure output is 2D [B, latent_dim]
        if z.ndim != 2 or z.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Encoder output shape mismatch: expected [B, D], got {z.shape}"
            )

        data.append(z)
        masks.append(mask)
        encode_masks.append(encode_mask)
        
    return data, masks, encode_masks

    
def process_batch(
        batch,
        vae_dictionary,
        sentinel_value,
        filtering = 'weights',
        verbose = True
    ):
    

    """
    Preprocess a batch by aligning dtypes/devices, extracting VAE latent representations,
    processing signals which are not encoded into the latent space.

    Parameters
    ----------
    batch : dict
        A batch dictionary with at least:
        - `batch['x']`: list[torch.Tensor] to contain the input tensors plus actuators,
        - `batch['y']`: list[torch.Tensor] to contain the output tensors,
        Tensors are expected to share the same batch size `B` on shape[0]. 
        
    vae_dictionary : dict[str, beta_VAE]
         vae_dictionary = {
            "input": dict[str, beta_VAE],
            "actuator": dict[str, beta_VAE] or [], 
            "output": dict[str, beta_VAE] or []
         }
    sentinel_value : float
        A place holder (99.0) that replaced NaN

    Returns
    -------
    input_data : torch.Tensor
        Concatenated latent representation for inputs/actuators.
    target_data : torch.Tensor
        Concatenated latent representation for outputs/targets.
    input_encode_mask : torch.Tensor
        Tensor of 1 (0) for valid (invalid) entries in `input`.
    valid_target : torch.Tensor
        Tensor of 1 (0) for valid (invalid) entries in `target_data`.
    """

    # Original data
    x = batch['x'] # Input + actuator
    y = batch['y'] # Output

    # Alignment and data transfer
    try:
        first_input_model = next(iter(vae_dictionary["input"].values()))
        p = next(first_input_model.parameters())
        input_  = [x_.to(dtype = p.dtype, device = p.device) for x_ in x] # real space data
        target_ = [y_.to(dtype = p.dtype, device = p.device) for y_ in y] # real space target
    except Exception as e:
        raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")
    
    sentinel = torch.as_tensor(
                sentinel_value,
                device = input_[0].device,
                dtype = input_[0].dtype
            )

    # When the signals are absent from the shot the MAST_tools sets them to NaN
    # We change them to sentinel_value as for the rest of the analysis.
    for i, inp in enumerate(input_):
        input_[i] = torch.nan_to_num(input_[i], nan=sentinel, posinf=sentinel, neginf=sentinel)
    for j, tar in enumerate(target_):
        target_[j] = torch.nan_to_num(target_[j], nan=sentinel, posinf=sentinel, neginf=sentinel)

    # Collect VAEs
    input_vae = list(vae_dictionary["input"].values()) + \
                list(vae_dictionary["actuator"].values())
    target_vae = list(vae_dictionary['output'].values())

    # Inputs are always encoded, hence encode_masks is used, see process_data method.
    input_data_list, _, input_encode_mask_list = process_data(input_vae, input_, sentinel) 

    # Targets can be either encoded or left in the real space in either cases we use the masks return 
    # value of process_data since this offers a mask per entry of each tensor.
    target_data_list, valid_target_list, _ = process_data(target_vae, target_, sentinel)

    input_data_cat = torch.cat(input_data_list, dim=1)
    target_data = torch.cat(target_data_list, dim=1)

    # if input_data_cat.ndim != 3 or input_encode_mask.ndim != 3 or target_data.ndim !=3:
    #     raise ValueError(f"Input data (mask) dimension expected to be 3, \
    #                     instead is {input_data_cat.ndim} ({input_encode_mask.ndim}) ")
                        
    # Get rid of poor batches
    input_encode_mask = torch.stack(input_encode_mask_list, dim=1)  # [B, n_signals]
    input_sample_completness = input_encode_mask.mean(dim=1)  # [B]
    if (input_sample_completness < 0.75).float().mean().item() > 0.95:
        if verbose:
            print("No valid latent space representation. Too many input signals are missing in this batch. ")
        return None, None, None, None, None
    
    # Concatenate encode mask to signal
    input_data = torch.cat([input_data_cat, input_encode_mask], dim=1)

    # Get rid of poor batches
    valid_target = torch.cat(valid_target_list, dim=1)
    target_sample_completness = (valid_target.mean(dim=1) < 0.75)
    if (target_sample_completness < 0.5).float().mean().item() > 0.50:
        if verbose:
            print("No valid target. Too many output signals are missing in this batch.")
        return None, None, None, None, None
    
    if filtering == 'hard_filtering':
        weights = None
        hard_filter = (input_sample_completness >= 0.75) & (target_sample_completness >= 0.5)

        # Filter data and mask
        input_data = input_data[hard_filter]
        valid_target = valid_target[hard_filter]
        target_data = target_data[hard_filter]
    else:
        weights = torch.minimum(input_sample_completness, target_sample_completness)
        hard_filter = None

    return input_data, target_data, input_encode_mask, valid_target, weights
    
    
def masked_loss(reco, target, mask, weights, eps = 1e-8):
    """
    Compute a mean squared error loss using an explicit validity mask.

    The loss is computed per sample over valid target entries only
    and then averaged over the batch.

    Args:
        reco (torch.Tensor): Reconstructed output tensor, shape [B, ...].
        target (torch.Tensor): Target tensor, shape [B, ...].
        mask (torch.Tensor): tensor, same shape as target, 1 (0) valid (invalid) entries.
        weights : (torch.Tensor) 
            for each sampe b in B, a weight is given that provides % of completness.
        eps (float, optional): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    if target.shape != mask.shape:  
        raise ValueError(
            f"target and valid_target must have the same shape, "
            f"got {target.shape} and {mask.shape}"
        )

    if target.ndim !=2:
        print(f"WARNING target ndim: expected 2 but got {target.ndim}")

    # Reduce over all non-batch dimensions
    dims = tuple(range(1, target.ndim))
    valid_per_sample = mask.sum(dim=dims) # nr. of valid entries per sample [B]

    squared_diff = mask * (target - reco)**2 # [B]
    sqr_sum_per_sample = squared_diff.sum(dim=dims) # [B]
    loss_per_sample = sqr_sum_per_sample/(valid_per_sample + eps) # loss per sample # [B]
    
    has_valid = valid_per_sample > 0

    # Compute weighted mean loss per batch only on valid samples
    if weights is not None:
        loss = (weights[has_valid] * loss_per_sample[has_valid]).sum() / weights[has_valid].sum()
    else:
        loss = loss_per_sample[has_valid].mean()

    return loss
