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
    
from src.pipelines.utils.layer_factory import SequentialBuilder

def build_conv1d_encoder_decoder(SETTINGS):
    """Build encoder and decoder for the conv1d_vae model.
 
    Parameters
    ----------
    SETTINGS : Settings
        Settings from the config.json.
    Returns
    -------
    dictionaries
        conv1d_encoder : SequentialBuilder or None
        conv1d_decoder : SequentialBuilder or None
        intermediate_layer_size :int, 
        conv1d_out_dim: int, 
        OR None if any error is encountered.     
    """
    # Initialize
    conv1d_encoder = None
    conv1d_decoder = None 
    intermediate_layer_size = None 
    conv1d_out_dim = None
        
    # Check if conv1d and beta_vae are available in SETTINGS
    if not hasattr(SETTINGS, "CONV1dENCODER"):
        print("Settings for conv1d NOT found: missing 'CONV1dENCODER'check 'conv1d_encoder' section in config")
        return None
    
    if not  hasattr(SETTINGS,"BETA_VAE"):
        print("Settings for conv1d NOT found: missing 'BETA_VAE, check 'beta-vae' section in config")
        return None

    if getattr(SETTINGS.BETA_VAE, "latent_dim", None) is None:
        print("Attribute latent_dim missing from SETTINGS.BETA_VAE")
        return None
    
    
    if not hasattr(SETTINGS, "ENCODER_SPECS") or not hasattr(SETTINGS.ENCODER_SPECS, "activation_fn"):
        print("Missing 'ENCODER_SPECS.activation_fn' in SETTINGS.")
        return None

    # Check all required attributes exist
    encoder_attributes = SETTINGS.CONV1dENCODER
    
    required_attrs = [
            "conv1d_in_channels",
            "conv1d_out_channels",
            "kernel",
            "stride",
            "padding"
            ]
    
    missing = [a for a in required_attrs
                if not hasattr(encoder_attributes, a) or getattr(encoder_attributes, a) is None]

    if missing:
        print(f"Missing keys in conv1d settings. Required keys {required_attrs}")
        return None
    
    # Create layer specs from SETTINGS. Retrieve nr. of channels and sig. length after each conv1d layers.
    encoder_specs, all_nr_channels, all_lengths = _conv1d_encoder_specs(SETTINGS)
    
    # Validate shapes returned
    if not all_nr_channels or not all_lengths:
        print("Encoder spec function returned empty channel/length lists.")
        return None

    # Find shape after conv1d-encoding
    conv1d_out_dim = all_nr_channels[-1] * all_lengths[-1]
    
    # Add Fully Connected Layer to encoder specs
    intermediate_layer_size = int( (conv1d_out_dim + SETTINGS.BETA_VAE.latent_dim)/2)
    FCL = {
        "type":"linear",
            "params": {
                "in_features": conv1d_out_dim,
                "out_features": intermediate_layer_size 
            }
        }

    conv1d_encoder_specs = copy.deepcopy(encoder_specs)  # independent clone # make copy before adding new layers
    encoder_specs = add(encoder_specs, FCL)
    encoder_specs = add(encoder_specs, {"type": SETTINGS.ENCODER_SPECS.activation_fn})
    
    # Build encoder from layers specs
    if encoder_specs is not None:
        conv1d_encoder = SequentialBuilder(encoder_specs)
    
    # Build decoder specs from conv1d_encoder_specs (NB: do not use encoder_specs)
    conv1d_decoder_specs = _build_decoder_specs_from_encoder_specs(
        SETTINGS, 
        conv1d_encoder_specs, 
        all_nr_channels,
        all_lengths,
        remove_last_activation = True)
    
    # Add fully connected layers
    
    FCL2 = {
        "type":"linear",
            "params": {
                "in_features": intermediate_layer_size,
                "out_features":  conv1d_out_dim
            }
        }
    
    FCL3 = {
        "type":"linear",
            "params": {
                "in_features": SETTINGS.BETA_VAE.latent_dim,
                "out_features": intermediate_layer_size    
            }
        }
    
    conv1d_decoder_specs = add({"type":SETTINGS.ENCODER_SPECS.activation_fn},conv1d_decoder_specs)
    conv1d_decoder_specs = add(FCL2, conv1d_decoder_specs)
    conv1d_decoder_specs = add({"type":SETTINGS.ENCODER_SPECS.activation_fn},conv1d_decoder_specs)
    conv1d_decoder_specs = add(FCL3, conv1d_decoder_specs)
    
    if conv1d_decoder_specs is not None:
        conv1d_decoder = SequentialBuilder(conv1d_decoder_specs)
        
    return conv1d_encoder, conv1d_decoder, intermediate_layer_size, conv1d_out_dim


def _conv1d_out_len(L_in, k, s=1, p=0, d=1):
    return math.floor((L_in + 2*p - d*(k-1) - 1) / s) + 1

def _compute_conv_output_dim(in_channels, input_length, layer_specs):
    """Compute output dimensions after Conv1d layers."""
    current_channels = in_channels
    current_length = input_length
    
    lengths = [input_length]
    channels = [in_channels]
    
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
    # L_out_target = desired output of convtranspose
    # L_in = input length to convtranspose (which equals encoder's L_out at that stage)
    base = (L_in - 1) * s - 2*p + d*(k-1) + 1
    op = L_out_target - base
    return op

def _build_decoder_specs_from_encoder_specs(SETTINGS, encoder_layer_specs, enc_c, enc_len,remove_last_activation):
    """Define decoder specs from the encoder specs. The decoder is a series of ConvTranspose1d
    with parametrization chosen to mirror the parameters in the encoder.

    Parameters
    ----------
    SETTINGS: Settings
        Settings from the config.json.
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
            decoder_layer_specs["layers"].append({"type":  SETTINGS.ENCODER_SPECS.activation_fn})
            i -= 1
            
    # For standardized targets, remove last relu 
    if remove_last_activation and decoder_layer_specs["layers"][-1]["type"] ==  SETTINGS.ENCODER_SPECS.activation_fn:
        decoder_layer_specs["layers"].pop()   
            
    return decoder_layer_specs

def _conv1d_encoder_specs(SETTINGS):
    """Create ecoder layers specs.

    Parameters
    ----------
    SETTINGS : Settings
        Configuration settings

    Returns
    -------
    list[dict]
        dictionary specifying encoder layers specs.

    """
    
    if not getattr(SETTINGS.TIME_SEGMENTATION, "targeted_time_stamps_per_window", None):
        print("Conv1d (time) input_length unresolved")
        return None

    input_length = SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window
    
    if not getattr(SETTINGS.CONV1dENCODER, "conv1d_in_channels", None):
        print("Conv1d conv1d_in_channels unresolved")
        return None
    
    in_channels = SETTINGS.CONV1dENCODER.conv1d_in_channels
        
    if not (len( SETTINGS.CONV1dENCODER.conv1d_out_channels) == len(SETTINGS.CONV1dENCODER.kernel) \
        == len(SETTINGS.CONV1dENCODER.stride) == len(SETTINGS.CONV1dENCODER.padding)):
        print(
            f"Mismatch in layer specs: "
            f"out_channels={len(out_channels_list)}, "
            f"kernel={len(kernel_list)}, "
            f"stride={len(stride_list)}, "
            f"padding={len(padding_list)}"
        )
        return None

    layers = []
    for out_channels, kernel, stride, padding in zip(
        SETTINGS.CONV1dENCODER.conv1d_out_channels,
        SETTINGS.CONV1dENCODER.kernel,
        SETTINGS.CONV1dENCODER.stride,
        SETTINGS.CONV1dENCODER.padding
    ):
        layers.append({
            "type": "conv1d",
            "params": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel,
                "stride": stride,
                "padding": padding
            },
        })
        layers.append({"type":   SETTINGS.ENCODER_SPECS.activation_fn})
        in_channels = out_channels
    
    # Check signal shape after stack of conv layers
    all_nr_channels, all_lengths = _compute_conv_output_dim(SETTINGS.CONV1dENCODER.conv1d_in_channels, input_length, {"layers": layers})
    
    if  all_lengths[-1] <=1:
        print("Signal after stack of conv1d has length <=1")
        return None
    
    # Make encoder specs dictionary
    conv1d_layer_specs = {"layers": layers}
    
    return conv1d_layer_specs, all_nr_channels, all_lengths

def add(l1, l2):
    
    """ Add l2 to l1 (or l1 to l2 when l2 is a container of layers).

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
                    "type": "relu"
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


def FullyConnectedLinearRelu(in_features, out_features):
    layer_specs = {
            "layers": [
                {
                    "type": "linear",
                    "params": {
                        "in_features": in_features,
                        "out_features": out_features
                    }
                },
                {
                    "type": "relu"
                }
            ]
        }
    return layer_specs

    
def main():
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
        
    from src.pipelines.configs.config_setup import get_settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        default = "src/pipelines/configs/config_flux_loop_flux.json",
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
        except Exception as e:
            print(f"Error in loading configuration {e}")
            return 
    
    return build_conv1d_encoder_decoder(SETTINGS)  

if __name__ == "__main__":
    conv1d_encoder, conv1d_decoder, intermediate_layer_size, conv1d_out_dim =  main()
    print(f"conv1d_encoder \n {conv1d_encoder}")
    print(f"conv1d_decoder \n {conv1d_decoder}")