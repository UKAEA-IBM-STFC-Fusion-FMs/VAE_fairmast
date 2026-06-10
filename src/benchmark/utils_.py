import argparse
import os
import torch
import sys
from typing import List, Optional


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
    models: List[Optional[beta_VAE]],
    batched_data: List[torch.Tensor]):
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
        List of batched tensors. Each entry corresponds to a signal in the sample `b` in the batch. 1
        For instance, the list could include input_signal_1, 2, ..., actuator_signal_1, .. etc; or output_signal_1, 2, etc.
        Each signal is expected to have the same batch size `B` on shape[0] and at least one additional dimension for the signal features.
    Returns
    -------
    data : list[torch.Tensor]
        List of batched tensors after processing. 
        The processing consits in two mutually exclusive options:

            1) The tensors in the batch are compressed in the latent space representation, if
            corresponding VAEs exist in models.

            2) The tensors in the batch are kept in their representations in the real space 
            but reshaped to agree with the dimension of compressed tensors, i.,e., 2D

    masks : list[torch.Tensor]
         It has the same shape of data. Each entry is a mask defined as follows:
       
        - If model is None:
            mask is elementwise (1 for valid entries, 0 for NaN/inf)

        - If model is not None:
            mask is sample-level (1 if sample is valid enough, 0 otherwise),
            broadcast across latent dimensions
    """
    # Sanity check
    if len(models) != len(batched_data):
        raise ValueError(
            f"Allignment problem: \
            the length of the list of models: ({len(models)}), is different \
            from the number of data given in the list of data: ({len(batched_data)})"
        )
        
    # Return values
    data: List[torch.Tensor] = []   # One entry per signal in batch_data 
    masks: List[torch.Tensor] = [] # One mask (entry) per signal in batch_data 

    # Encode or reshape tensors
    for model, batch in zip(models, batched_data):
        
        # Check nr. of dimensions
        if batch.ndim < 2:
            raise ValueError("Each batch must have at least 2 dimensions [B, ...]")

        # Batch size
        B = batch.shape[0]
        dims = tuple(range(1, batch.ndim))
        
        # Mask invalid entries (NaN or inf) in the batch
        invalid_entries = torch.isnan(batch) | torch.isinf(batch)
        
        # Ground invalid entries with zero
        x = batch.clone()
        x[invalid_entries] = 0
        
        # Check data for encoding
        if model is not None:
        
            p = next(model.parameters())
            if batch.device != p.device or batch.dtype != p.dtype:
                raise ValueError(f"Batch is not on the same device as the model: {p.device}")
            
            
            # Initialize mask and latent representation with zeros
            z = torch.zeros(B, model.latent_dim, device=batch.device, dtype=batch.dtype)
            mask = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            
            # Invalidate poor quality samples based on the fraction of invalid entries in the original batch
            invalid_fraction_per_sample = invalid_entries.float().mean(dim=dims)  # [B]
            tau = 0.05 
            valid_samples = invalid_fraction_per_sample <= tau  # bool [B]

            if valid_samples.any():
                with torch.no_grad():
                    z[valid_samples] = model.encode(x[valid_samples])[0]
                mask[valid_samples, :]  = 1
                mask = mask.to(dtype=batch.dtype)   
        else:
            z = batch.reshape(B, -1).clone()
            mask = (~(torch.isnan(z)|torch.isinf(z))).to(dtype=batch.dtype)
            z = z * mask

        # Ensure output is 2D [B, latent_dim]
        if z.ndim != 2 or z.shape[0] != batch.shape[0]:
            raise ValueError(
                f"Encoder output shape mismatch: expected [B, D], got {z.shape}"
            )

        data.append(z)
        masks.append(mask)
        
    return data, masks

    
def process_batch(
        batch,
        vae_dictionary,
        filtering = 'weights'
    ):
    
    """
    Preprocess a batch by:
    1) Aligning dtype/device with the reference VAE.
    2) Encoding signals with VAEs when available.
    3) Flattening signals without VAEs to 2D.
    4) Concatenating data with corresponding masks.

    Parameters
    ----------
    batch : dict
        Must contain:
        - 'x': list[Tensor] (inputs + actuators)
        - 'y': list[Tensor] (targets)
        All tensors must share the same batch size B.

    vae_dictionary : dict
        {
            "input": dict[str, beta_VAE],
            "actuator": dict[str, beta_VAE] or {},
            "output": dict[str, beta_VAE] or {}
        }

    Returns
    -------
    input_data : Tensor [B, D_in + n_in]
        Concatenated encoded inputs with sample-level masks.

    target_data : Tensor [B, D_out + n_out]
        Concatenated encoded targets with sample-level masks.
        
    target_mask : Tensor [B, n_out]
        Sample-level mask for the target data (1 if valid, 0 if invalid).

    weights : Tensor [B] or None
        Sample-wise weights (soft filtering) or None if hard filtering is used.
    """

    # Original data
    x = batch['x'] # Input + actuator
    y = batch['y'] # Output

    # Alignment and data transfer
    if len(vae_dictionary["input"]) == 0:
        raise ValueError("Input VAE is required.")

    try:
        first_input_model = next(iter(vae_dictionary["input"].values()))
        p = next(first_input_model.parameters())
        input_  = [x_.to(dtype = p.dtype, device = p.device) for x_ in x] # real space data
        target_ = [y_.to(dtype = p.dtype, device = p.device) for y_ in y] # real space target
    except Exception as e:
        raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")

    # Collect VAEs
    actuator_dict = vae_dictionary["actuator"] or {}
    output_dict = vae_dictionary["output"] or {}

    input_vae = list(vae_dictionary["input"].values()) + list(actuator_dict.values())
    target_vae = list(output_dict.values())
    
    if len(input_vae) != len(input_):
        raise ValueError("Mismatch between number of input tensors and VAEs")

    # Inputs are always encoded, hence encode_masks is used, see process_data method.
    input_data_list, input_mask_list = process_data(input_vae, input_) 
    target_data_list, target_mask_list = process_data(target_vae, target_)

    input_data_cat = torch.cat(input_data_list, dim=1) # [B, sum n_signals]
    target_data = torch.cat(target_data_list, dim=1)  # [B, sum n_signals]
 
    input_mask = torch.cat(input_mask_list, dim=1)  # [B, sum n_signals]
    input_data = torch.cat([input_data_cat, input_mask], dim=1)  # [B, 2 * sum n_signals]

    target_mask = torch.cat(target_mask_list, dim=1)  # [B, sum n_signals]
    
    # Sample-level completeness
    input_sample_completness = input_mask.mean(dim=1)  # [B]
    target_sample_completness = target_mask.mean(dim=1)  # [B]
    
    if filtering == 'hard_filtering':
        weights = None
        hard_filter = (input_sample_completness >= 0.75) & (target_sample_completness >= 0.5)

        # Filter data and mask
        return input_data[hard_filter],  target_data[hard_filter], target_mask[hard_filter], None
    else:
        weights = torch.minimum(input_sample_completness, target_sample_completness)
        return input_data, target_data, target_mask, weights
    
    
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
