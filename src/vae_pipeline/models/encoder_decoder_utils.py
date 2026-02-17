"""Utilities to build encoder and decoder architectures for the beta-VAE model.

These methods rely on the settings specified in config json.
The keys and values in config json are store in the object SETTINGS. 
If some keys are missing from SETTINGS, these methods do not work, see _chek_for_missing_attributes method.

- build_conv1d_encoder_decoder
    This method builds the encoder and decoder by using:
        - Stack of nn.conv1d for the encoder.
        - Stack of nn.ConvTranspose1d for the decoder.
    The encoder specs, i.e., layers and correspondig parameters such as: kernel, stride and padding,
    are read from config json. The decoder config is either passed via the config file or built automatically
    to mirror the encoder specs (see _build_convTransp_decoder_specs_from_encoder_specs)

- _build_convTransp_decoder_specs_from_encoder_specs
    This method builds the decoder specs, i.e., layers and correspondig parameters such as: kernel, stride and padding,
    by mirroring the setting of the encoder.
    
"""
import copy
import os
import math
import sys
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
        ".."
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from src.vae_pipeline.utils.layer_factory import SequentialBuilder

def active_linear(in_features, out_features, activation_fn):
    return  {
            "layers": [
                {
                    "type": "linear",
                    "params": {
                        "in_features": in_features,
                        "out_features": out_features
                    }
                },
                {
                    "type": activation_fn
                }
            ]
        }

def add(l1, l2):
    
    """ 
    - Add l2 to l1, if l1 is a list of layers and l2 is a single layer or list of layers.
    - Add l1 to la2, if l2 is a list of layers and l1 is a single layer.
 

    Parameters
    ----------
    l1, l2 : dict
    
    Either
        
        {
            "layers": [
                {
                    "type": "linear",
                    "params": {
                        "in_features": in_features,
                        "out_features": out_features
                    }
                },
                {
                    "type": "activation_fn"
                }
            ]
        }
 
    OR 
        
        {
            "type": "linear",
            "params": {
                "in_features": in_features,
                "out_features": out_features
            }
        },
    
    Returns
        -------
        dict
            The updated structure after adding.
    """

    
    if "layers" in l1 and isinstance(l1["layers"], list):
        if isinstance(l2, dict) and "type" in l2 and "layers" not in l2:
            
            l1["layers"].append(l2)
            return l1

    if "layers" in l2 and isinstance(l2["layers"], list):
        if isinstance(l1, dict) and "type" in l1 and "layers" not in l1:
            
            l2["layers"].insert(0, l1)
            return l2
        
    if "layers" in l1 and isinstance(l1["layers"], list):
        if "layers" in l2 and isinstance(l2["layers"], list):
            l1["layers"].extend(l2["layers"])
            return l1 

def quick_build_from_config(SETTINGS):
    if _chek_for_missing_attributes(SETTINGS):
        return None
    
    # Get encoder specs
    encoder_specs = {"layers": SETTINGS.ENCODER.layers}
    
    # Build encoder from layers specs
    if encoder_specs["layers"]:
        conv1d_encoder = SequentialBuilder(encoder_specs)
    
    # Read decoder specs from SETTINGS is decoder key is available
    if SETTINGS.get("DECODER","layers"):
        decoder_specs = {"layers": SETTINGS.DECODER.layers}
    else:
        raise KeyError("Decoder specs not available in config file")
    
    # Build decoder from layers specs
    if decoder_specs["layers"]:
        conv1d_decoder = SequentialBuilder(decoder_specs)

    # Ger the size of last linear layer
    out_layer_size = None
    for spec in reversed(encoder_specs["layers"]):
        if spec["type"] == "linear": 
            out_layer_size = spec["params"]["out_features"]
            break

    return conv1d_encoder, conv1d_decoder, out_layer_size
        
    
def build_conv1d_encoder_decoder(SETTINGS):
    """Build encoder and decoder for the conv1d_vae model. 
    The encoder specs are stored in the SETTINGS. 
    The decoder specs are either stored in SETTINGS or they are built automatically
    by mirroring the encoder specs.

    ********* IMPORTANT *****************
            The automatic building of the decoder from encoder specs only works for encoders built of
            only conv1d and activation_fn layers, for example:
            {"type": "conv1d", "params": {"in_channels": 15, "out_channels": 64, "kernel_size": 5, "stride": 2, "padding": 0}}, 
            {"type": "activation_fn"}, 
            {"type": "conv1d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 0}}, 
            {"type": "activation_fn"}, 
            {"type": "conv1d", "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 4, "stride": 1, "padding": 0}}, 
            {"type": "activation_fn"}
 
    Parameters
    ----------
    SETTINGS : Object
        Settings obtained from the config.json.
    Returns
    -------
    dictionaries
        conv1d_encoder : SequentialBuilder or None
        conv1d_decoder : SequentialBuilder or None
        size_before_vae :int, 
        conv1d_out_dim: int, 
        OR None if any error is encountered.     
    """
    # Initialize
    conv1d_encoder = None
    conv1d_decoder = None 
    size_before_vae = None 
    conv1d_out_dim = None

    if _chek_for_missing_attributes(SETTINGS):
        return None
    
    # Create layer specs from SETTINGS. Retrieve nr. of channels and sig. length after each conv1d layers.
    encoder_specs = {"layers": SETTINGS.ENCODER.layers}
    conv1d_encoder_specs = copy.deepcopy(encoder_specs)  # independent clone # make copy before adding new layers

    # Get signal shape after conv1d encoder
    all_nr_channels, all_lengths = _compute_conv_output_dim(SETTINGS, encoder_specs)
    conv1d_out_dim = all_nr_channels[-1] * all_lengths[-1]
    
    # Size of intermediate layer (if any) before latent space representation
    size_before_vae = int( (conv1d_out_dim + SETTINGS.BETA_VAE.latent_dim)/2)
    #size_before_vae = conv1d_out_dim
    
    # Add layers to encoder specs    
    encoder_specs = add(encoder_specs, {"type": "flatten", "params":{"start_dim":1}})
    L1 = active_linear(in_features = conv1d_out_dim, out_features = size_before_vae, activation_fn = SETTINGS.ENCODER.activation_fn)
    encoder_specs = add(encoder_specs, L1)
    
    # Build encoder from layers specs
    if encoder_specs is not None:
        conv1d_encoder = SequentialBuilder(encoder_specs)

    # Read decoder specs from SETTINGS is decoder key is available
    if SETTINGS.get("DECODER","layers"):
        conv1d_decoder_specs = {"layers": SETTINGS.DECODER.layers}
    else:
        # Build decoder specs from conv1d_encoder_specs (NB: do not use encoder_specs)
        conv1d_decoder_specs = _build_convTransp_decoder_specs_from_encoder_specs(
            SETTINGS, 
            conv1d_encoder_specs, 
            all_nr_channels,
            all_lengths,
            remove_last_activation = True)
        
        conv1d_decoder_specs = add(
            {
                "type": "unflatten",
                "params": {
                    "dim":1, 
                    "unflattened_size":(all_nr_channels[-1], all_lengths[-1])
                    }
            },
            conv1d_decoder_specs
            )
        
        # Add fully connected layers
        L2 = active_linear(in_features = size_before_vae, out_features = conv1d_out_dim, activation_fn = SETTINGS.ENCODER.activation_fn) 
        L3 = active_linear(in_features = SETTINGS.BETA_VAE.latent_dim, out_features = size_before_vae, activation_fn = SETTINGS.ENCODER.activation_fn) 

        conv1d_decoder_specs = add(L2, conv1d_decoder_specs)
        conv1d_decoder_specs = add(L3, conv1d_decoder_specs)
    
    if conv1d_decoder_specs is not None:
        conv1d_decoder = SequentialBuilder(conv1d_decoder_specs)

    return conv1d_encoder, conv1d_decoder, size_before_vae

def _build_convTransp_decoder_specs_from_encoder_specs(SETTINGS, encoder_layer_specs, enc_c, enc_len,remove_last_activation):
    """Define decoder specs from the encoder specs. The decoder is a series of ConvTranspose1d
    with parametrization chosen to mirror the parameters in the encoder.

    Parameters
    ----------
    SETTINGS: object
        Settings obtained from the config.json.
    encoder_layer_specs : list[dict]
        dictionary specifying encoder layers specs.
    enc_c : list[int]
        the i-th entry is the in_channel parameter of the i-th conv1d layer in the encoder.
    enc_len : list[int] 
        the i-th entry is the lenght of the output signal after the i-th conv1d layer in the encoder.  

    Returns
    -------
    decoder_layer_specs : list[dict]
    dictionary specifying decoder layers specs.
    """
    
    print(f"encoded layers length {enc_len}")
    
    # Pass 2: build decoder layers in reverse
    decoder_layer_specs = {"layers": []}

    # Reverse loop over encoder layers    
    i=len(enc_len)-1

    for spec in reversed(encoder_layer_specs["layers"]):
        if spec["type"] != SETTINGS.ENCODER.activation_fn and spec["type"] != "conv1d":
            raise ValueError(
                "Encoder architecture is different from what is expected. "
                "Automatic building of decoder specs from encoder specs only works "
                "for certain encoder architectures. "
                "For more complex architectures, pass the decoder specs via SETTINGS "
                "by using the config JSON file."
            )
        if spec["type"] == "conv1d":
            params = spec.get("params", {})
            k = params.get("kernel_size", 1)
            s = params.get("stride", 1)
            p = params.get("padding", 0)
            d = params.get("dilation", 1)
           
            in_len = enc_len[i] # current length
            target_len = enc_len[i-1] # we want to get back to this length

            # op: output_padding is additional size added to one side of the output shape
            # It is only used to find output shape, but does not actually add zero-padding to output.
            op = _convt1d_needed_output_padding(in_len, target_len, k, s, p, d)
            if not (0 <= op <= s - 1):
                op = max(0, min(s - 1, op))
                
            spec = {
                "type": "conv_transpose1d",
                "params": {
                    "in_channels": enc_c[i],
                    "out_channels": enc_c[i-1],
                    "kernel_size":k, 
                    "stride": s,
                    "padding": p,
                    "dilation": d,
                    "output_padding": op
                }
            }
            decoder_layer_specs["layers"].append(spec)
            # Add ReLU spec after each ConvTranspose1d
            decoder_layer_specs["layers"].append({"type":  SETTINGS.ENCODER.activation_fn})
            i -= 1
            
    # For standardized targets, remove last activation_fn 
    if remove_last_activation and decoder_layer_specs["layers"][-1]["type"] ==  SETTINGS.ENCODER.activation_fn:
        decoder_layer_specs["layers"].pop()   
            
    return decoder_layer_specs

def build_linear_encoder_decoder(SETTINGS):
    """
    Build linear encoder/decoder from SETTINGS.

    Creates an encoder from `SETTINGS.ENCODER.layers`. The decoder is built either 
    via `SETTINGS.DECODER.layers`or by mirroring the encoder's linear layers 
    (reverse order, swap in/out features).
    The decoder also has an extra linear layer that joins the latent space representation
    to the output space from the encoder:
    active_linear(SETTINGS.BETA_VAE.latent_dim -> out_dim, activation_fn)`.

                    ********* IMPORTANT *****************
                    The automatic building of the decoder from encoder specs only works for encoders built of
                    only linear and SETTINGS.ENCODER.activation_fn layers, for example:
                    {"type": "linear", "params": {"in_features": 20, "out_features": 80}}, 
                    {"type": "activation_fn"},
                    {"type": "linear", "params": {"in_features": 80, "out_features": 40}}, 
                    {"type": "activation_fn"}
                    
    Parameters
    ----------
    SETTINGS : object
        Settings obtained from the config.json with:
        - ENCODER.layers : list[dict] (linear specs with in_features/out_features)
        - ENCODER.activation_fn : str
        - DECODER.layers : list[dict], optional
        - BETA_VAE.latent_dim : int

    Returns
    -------
    tuple
        (linear_encoder, linear_decoder, out_dim)
        where `out_dim` is the `out_features` of the last encoder linear layer.
    Notes
    -----
    Returns None early if `_chek_for_missing_attributes(SETTINGS)` is truthy.
    Only `"type" == "linear"` layers are mirrored.
    
    """


    # Initialize
    linear_encoder = None
    linear_decoder = None 
    out_dim = None

    if _chek_for_missing_attributes(SETTINGS):
        return None
    
    # Create layer specs from SETTINGS. Retrieve nr. of channels and sig. length after each conv1d layers.
    encoder_specs = {"layers": SETTINGS.ENCODER.layers}
    linear_encoder = SequentialBuilder(encoder_specs)

    # Read decoder specs from SETTINGS is decoder key is available
    if SETTINGS.get("DECODER","layers"):
        decoder_specs = {"layers": SETTINGS.DECODER.layers}
        
        for spec in reversed(encoder_specs["layers"]):
            if spec["type"] == "linear":
                
                # Retrieve out_dim of linear layer of the encoder
                out_features = spec["params"]["out_features"]
                in_features = spec["params"]["in_features"]
                
                if out_dim is None:
                    out_dim = out_features
                    
    else:
        # Build decoder by mirroring encoder
        decoder_specs = {"layers": []}
        for spec in reversed(encoder_specs["layers"]):
            if spec["type"]!= SETTINGS.ENCODER.activation_fn and spec["type"]!= "linear":
                raise ValueError(
                    "Encoder architecture is different from what is expected. "
                    "Automatic building of decoder specs from encoder specs only works "
                    "for certain encoder architectures. "
                    "For more complex architectures, pass the decoder specs via SETTINGS "
                    "by using the config JSON file."
                )
            if spec["type"] == "linear":
                
                # Retrieve out_dim of linear layer of the encoder
                out_features = spec["params"]["out_features"]
                in_features = spec["params"]["in_features"]
                
                if out_dim is None:
                    out_dim = out_features

                decoder_specs["layers"].append(
                    {"type": "linear",
                    "params":{
                        "in_features":out_features,
                        "out_features":in_features
                            }
                        }   
                    )
                decoder_specs["layers"].append({"type":SETTINGS.ENCODER.activation_fn})
                
        L1 = active_linear(in_features =  SETTINGS.BETA_VAE.latent_dim, out_features =out_dim, activation_fn = SETTINGS.ENCODER.activation_fn)
        decoder_specs = add(L1, decoder_specs)

    if decoder_specs is not None:
        linear_decoder = SequentialBuilder(decoder_specs)  
        
    return linear_encoder, linear_decoder, out_dim         

def _chek_for_missing_attributes(SETTINGS):
    """
    Validate that all required configuration attributes are present in SETTINGS.

    Checks for mandatory fields used by encoder/decoder construction, including
    time segmentation parameters, latent dimension, encoder type and activation
    function, and (if present) Conv1D encoder settings. Prints diagnostic messages
    for missing attributes.

    Returns
    -------
    bool
        True if any required attribute is missing, otherwise False.
    """

    # Check if "targeted_time_stamps_per_window" exists and is not None in SETTINGS.TIME_SEGMENTATION
    if SETTINGS.get("TIME_SEGMENTATION", "targeted_time_stamps_per_window") is None:
        print("Conv1d (time) input_length unresolved")
        return True

    # Check if "latent_dim" exists and is not None in SETTINGS.BETA_VAE
    if SETTINGS.get("BETA_VAE", "latent_dim") is None:
        print("Attribute latent_dim missing from SETTINGS.BETA_VAE")
        return True

    if SETTINGS.get("ENCODER", "activation_fn") is None:
        print("Attribute activation_fn missing from SETTINGS.ENCODER")
        return True
    
    if SETTINGS.get("ENCODER", "type") is None:
        print("Attribute type missing from SETTINGS.ENCODER")
        return True
    
    # Check all required attributes exist
    if hasattr(SETTINGS,"CONV1dENCODER"):
        print("'conv1d_encoder' was found in config file.")
        required_attrs = [
                "conv1d_in_channels",
                "conv1d_out_channels",
                "kernel",
                "stride",
                "padding"
                ]
        
        missing = [a for a in required_attrs if SETTINGS.get("CONV1dENCODER", a) is None]

        if missing:
            print(f"Missing keys in conv1d settings. Required keys {required_attrs}")
            return True
    
    return False

def _compute_conv_output_dim(SETTINGS, layer_specs):
    """Compute output dimensions after conv1d layers configured in layer_specs."""
    
    current_channels = layer_specs["layers"][0]["params"]["in_channels"]
    current_length =  SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window
    
    lengths = [current_length]
    channels = [current_channels]
    
    for spec in layer_specs["layers"]:
        if spec["type"] == "conv1d":
            params = spec.get("params", {})
            kernel_size = params.get("kernel_size", 1)
            stride = params.get("stride", 1)
            padding = params.get("padding", 0)
            out_channels = params.get("out_channels", current_channels)

            current_length =  _conv1d_out_len(current_length, kernel_size, stride, padding)
            current_channels = out_channels
            
            lengths.append(current_length)
            channels.append(current_channels)

    return channels, lengths

def _convt1d_needed_output_padding(L_in, L_out_target, k, s=1, p=0, d=1):
    """_summary_

    Parameters
    ----------
    L_in : int
        input length to convtranspose (which equals encoder's L_out at that stage)
    L_out_target : int
        desired output of convtranspose
    k : int
        kernel
    s : int, optional
        stride, by default 1
    p : int, optional
        padding, by default 0
    d : int, optional
        dilation, by default 1

    Returns
    -------
    int
        difference between traget and base length
    """
    base = (L_in - 1) * s - 2*p + d*(k-1) + 1
    op = L_out_target - base
    return op

def _conv1d_out_len(L_in, k, s=1, p=0, d=1):
    return math.floor((L_in + 2*p - d*(k-1) - 1) / s) + 1

       
if __name__ == "__main__":
    
    import argparse
    import os
    import sys
    
    REPO_ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
            "..",
            "..",
            ".."
        )
    )
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
        
    from src.vae_pipeline.configs.config_setup import get_settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "src/vae_pipeline/configs/config_solenoid_current_linear_encoder.json",
        type=str,
        help="Path to configuration file for the pipeline.")
    
    args = parser.parse_args()
    
    config_file_path = args.config_file_path
    config_file_name = os.path.basename(config_file_path)
    
    # Load configuration from JSON file
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.") 
    else:
        try:
            SETTINGS = get_settings(config_file_path) 
            try:
                conv1d_encoder, conv1d_decoder, size_before_vae, conv1d_out_dim =  build_conv1d_encoder_decoder(SETTINGS) 
                print(f"conv1d_encoder \n {conv1d_encoder}")
                print(f"conv1d_decoder \n {conv1d_decoder}")
            except Exception as e:
                print(f"Exception {e}")
            try:
                linear_encoder, linear_decoder, out_dim = build_linear_encoder_decoder(SETTINGS) 
                print(f"linear_encoder \n {linear_encoder}")
                print(f"linear_decoder \n {linear_decoder}")
            except Exception as e:
                print(f"Exception {e}")
            
        except Exception as e:
            print(f"Error in loading configuration {e}")

        
    