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


def load_vae_model(config_path:str, signal_name: str, device):
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
    
    
    parent_dir = os.path.dirname(config_path)
    best_model_path = os.path.join(parent_dir, "best_vae_" + signal_name + ".pt")
    
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    return model
    

def create_vae_dictionary(device, vae_dictionary, type_of_signal, list_of_signals, SETTINGS, model_sub_paths):
    """ Loads VAE models into a dictionary for signals in the list.

       Reinforce the one-to-one correspondence between input or actuator signals and VAE models.

    Parameters
    ----------
    mode: string
        eval or train
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

    for source, signal_name in list_of_signals:
        key = f"{source}-{signal_name}"
        vae_dictionary[type_of_signal][key] = None

        for model_path in model_sub_paths:
            if signal_name in model_path:
                
                full_path = os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, model_path)
                vae_dictionary[type_of_signal][key] = load_vae_model(full_path, signal_name, device)

    num_inputs = len(list_of_signals)
    num_loaded = len(vae_dictionary[type_of_signal])

    if num_inputs != num_loaded:
        if type_of_signal in ["input", "actuator"]:
            raise ValueError(
                f"The number of input signals ({num_inputs}) does not match the number of VAE "
                f"models loaded ({num_loaded})."
            )

    return vae_dictionary
    
def skim_batch(input_, target_):
    
    """
    Return a boolean mask indicating which batch samples should be kept.

    A sample is considered valid if more than half of its values are finite.

    Then, for each batch sample:
      - the fraction of valid entries across `input_` features is computed;
      - the fraction of valid entries across `target_` features is computed.

    A batch sample is discarded only if fewer than half of the input entries
    are valid OR fewer than half of the target entries are valid.
    Otherwise, it is kept.

    Args:
        input_ (list[torch.Tensor]):
            List of input tensors. Each tensor must have shape `[B, ...]`,
            where the batch dimension `B` is shared across all tensors.
        target_ (list[torch.Tensor]):
            List of target tensors. Each tensor must have shape `[B, ...]`,
            where the batch dimension `B` is shared across all tensors.

    Returns:
        torch.Tensor:
            Boolean tensor of shape `[B]`, where `True` means the
            corresponding batch sample should be kept and `False` means it
            should be discarded.
    """
    
    valid_input_entries = torch.stack(
        [(torch.isfinite(x_).float().mean(dim=tuple(range(1, x_.ndim))) > 0.5).float()
        for x_ in input_],
        dim=1
    )  # [B, n_input_entries]

    valid_target_entries = torch.stack(
        [(torch.isfinite(y_).float().mean(dim=tuple(range(1, y_.ndim))) > 0.5).float()
        for y_ in target_],
        dim=1
    )  # [B, n_target_entries]

    
    input_validity = valid_input_entries.mean(dim=1)   # [B], fraction of valid input entries
    target_validity = valid_target_entries.mean(dim=1) # [B], fraction of valid target entries

    valid_samples = ~((input_validity < 0.5) | (target_validity < 0.5))  # [B] boolean mask
    
    return valid_samples
    
def process_data(
    models: List[Optional[beta_VAE]],
    batched_data: List[torch.Tensor],
    mask_type):
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
        List of batched tensors. Each entry corresponds to a signal in the sample `b` in the batch. 
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
        finite_mask = torch.isfinite(batch)
        invalid_entries = ~finite_mask

        # Ground invalid entries with zero
        batch[invalid_entries] = 0
        
        # Invalidate poor quality samples based on the fraction of invalid entries in the original batch
        invalid_fraction_per_sample = invalid_entries.float().mean(dim=dims)  # [B]
        tau = 0.25
        valid_samples = invalid_fraction_per_sample <= tau  # bool [B]
            
        # Check data for encoding
        if model is not None:
        
            p = next(model.parameters())
            if batch.device != p.device or batch.dtype != p.dtype:
                raise ValueError(f"Batch is not on the same device as the model: {p.device}")
            
            # Initialize mask and latent representation with zeros
            z = torch.zeros(B, model.latent_dim, device=batch.device, dtype=batch.dtype)
            if mask_type  == "expand_mask_over_latent_dim": 
                mask = torch.zeros(B, model.latent_dim, device=batch.device, dtype = batch.dtype)
            else:
                mask = torch.zeros(B, 1, device=batch.device, dtype = batch.dtype)

            if valid_samples.any():
                with torch.no_grad():
                    z[valid_samples] = model.encode(batch[valid_samples])[0]
                mask[valid_samples, :]  = 1
                mask = mask.to(dtype=batch.dtype)   
        else:
            z = batch.reshape(B, -1)
            mask = finite_mask.reshape(B, -1).to(dtype=batch.dtype)
            z = z * mask

        data.append(z)
        masks.append(mask)
        
    return data, masks

    
def process_batch(
        batch,
        input_vae,
        target_vae
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

    input(target)_vae : list[dic]
        dic =  {
            "input": dict[str, beta_VAE],
            "actuator": dict[str, beta_VAE] or {},
            "output": dict[str, beta_VAE] or {}
        }

    Returns
    -------
    input_data : Tensor [B, D_in + n_in]
        Encoded inputs concatenated with sample- or element wise- level masks (1 if valid, 0 if invalid).

    target_data : Tensor [B, D_out + n_out]
        CEncoded/real targets.
    
    input_mask : Tensor [B, n_out]
        sample- or element wise- mask for the input data (1 if valid, 0 if invalid).
        
    target_mask : Tensor [B, n_out]
        sample- or element wise- mask for the target data (1 if valid, 0 if invalid).

    target_ : list(torch.Tensor)
        Each entry is a batch from the dataloader corresponding to one target signal.
        A batch is skimmed to retain only valid samples.
        
    input_ : list(torch.Tensor)
        Each entry is a batch from the dataloader corresponding to one input signal.
        A batch is skimmed to retain only valid samples.
    
    """

    # Original data
    x = batch['x'] # Input + actuator
    y = batch['y'] # Output

    first_input_model = input_vae[0]

    try:
        p = next(first_input_model.parameters())

        input_ = [
            x_ if (x_.device == p.device and x_.dtype == p.dtype) else x_.to(device=p.device, dtype=p.dtype)
            for x_ in x
        ]

        target_ = [
            y_ if (y_.device == p.device and y_.dtype == p.dtype) else y_.to(device=p.device, dtype=p.dtype)
            for y_ in y
        ]
    except Exception as e:
        raise ValueError(f"Error while aligning batch tensors with model dtype/device: {e}")
    
    valid_samples = skim_batch(input_, target_).to(input_[0].device)
    
    input_  = [x_[valid_samples] for x_ in input_]
    target_ = [y_[valid_samples] for y_ in target_]
    
    target_original = [t.clone() for t in target_]
    input_original =  [t.clone() for t in input_]
    
    # Process data
    input_data_list, input_mask_list = process_data(input_vae, input_, "do_not_expand_mask_over_latent_dim") 
    target_data_list, target_mask_list = process_data(target_vae, target_, "expand_mask_over_latent_dim")
    
    input_data = torch.cat(input_data_list, dim=1)  # [B, sum n_signals]
    input_mask = torch.cat(input_mask_list, dim=1)  # [B, sum n_signals]
    
    target_data = torch.cat(target_data_list, dim=1)  # [B, sum n_signals]
    target_mask = torch.cat(target_mask_list, dim=1)  # [B, sum n_signals]
    
    return input_data, target_data, input_mask, target_mask, target_original, input_original
    
    
def masked_loss(reco, target, mask, eps = 1e-8):
    """
    
    Compute a masked mean squared error over all valid entries.

    The loss is computed across all elements in the batch where mask == 1,
    ensuring each valid entry contributes equally.

    Args:
        reco (torch.Tensor): Reconstructed output tensor, shape [B, ...].
        target (torch.Tensor): Target tensor, shape [B, ...].
        mask (torch.Tensor): tensor, same shape as target, 1 (0) valid (invalid) entries.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    
    if target.shape != mask.shape:  
        raise ValueError(
            f"target and valid_target must have the same shape, "
            f"got {target.shape} and {mask.shape}"
        )
        
    
    if target.shape != reco.shape:
        raise ValueError(
            f"target and reco must have the same shape, "
            f"got {target.shape} and {reco.shape}"
        )

    if target.ndim !=2:
         raise ValueError(f"ERROR: target dimension must be 2. Current dim is {target.ndim}")
    
    sum_mask = mask.sum()
    if sum_mask > 0:
        loss = (mask * (target - reco)**2).sum() / sum_mask
    else:
        loss  = torch.tensor(0.0, device=target.device, dtype=target.dtype)
        
    return loss
