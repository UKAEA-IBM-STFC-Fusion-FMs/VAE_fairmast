import math
import torch
import torch.nn as nn


def build_conv1d_encoder_decoder(SETTINGS):
    """Build encoder and decoder specs for the conv1d model.
 
    Parameters
    ----------
    SETTINGS : Settings
        settings from config.json.
    Returns
    -------
    dictionaries
        encoder specs as dictionary, 
        decoder specs as dictionary.
    OR 
    None if any error is encountered
        
    """
    conv1d_encoder = None
    conv1d_decoder None 
        
    # Check if conv1d and beta_vae are available in SETTINGS
    if not hasattr(SETTINGS, "CONV1dENCODER"):
        print("Settings for conv1d NOT found: missing 'conv1d_encoder' section in config")
        return None
    
    if not  hasattr(SETTINGS,"BETA_VAE"):
        print("Settings for conv1d NOT found: missing 'beta_vae' section in config")
        return None

    if not getattr(SETTINGS.BETA_VAE, "latent_dim", None):
        print("Attribute latent_dim missing from SETTINGS.BETA_VAE")
        return None
    
    
    # Check all attributes exist
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
        print(f"Missing keys in conv1d settings. Required keys {required_keys}")
        return None
    
    # Create layer specs from SETTINGS. Retrieve nr. of channels and sig. length after each conv1d layer.
    encoder_layer_specs, all_nr_channels, all_lengths = _conv1d_encoder_specs(SETTINGS)
    
    # Find shape after conv-encoding
    conv_out_dim = all_nr_channels[-1] * all_lengths[-1]
    
    # Add Fully Connected Layer to encoder
    FCL = {
        "type":"linear",
            "params": {
                "in_features": conv_out_dim,
                "out_features": int( (conv_out_dim + SETTINGS.BETA_VAE.latent_dim)/2)
            }
        }
    
    conv1d_specs = encoder_layer_specs # make copy before adding new layers
    encoder_layer_specs = add_layer(encoder_layer_specs, FCL)
    encoder_layer_specs = add_layer(encoder_layer_specs, {"type": "relu"})
    
    # Build encoder from layers specs
    if encoder_layer_specs is not None:
        conv1d_encoder = SequentialBuilder(encoder_layer_specs)
    
    # Build decoder specs 
    conv1d_decoder_specs = _build_decoder_specs_from_encoder_specs(
        SETTINGS, 
        conv1d_specs, 
        all_nr_channels,
        all_lengths,
        remove_last_activation = True)
    
    
    return conv1d_encoder, conv1d_decoder


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

    return lengths, channels

def _convt1d_needed_output_padding(L_in, L_out_target, k, s=1, p=0, d=1):
    # L_out_target = desired output of convtranspose
    # L_in = input length to convtranspose (which equals encoder's L_out at that stage)
    base = (L_in - 1) * s - 2*p + d*(k-1) + 1
    op = L_out_target - base
    return op

def _build_decoder_specs_from_encoder_specs(SETTINGS, encoder_layer_specs, enc_c, enc_len):
    """Define decoder specs from the encoder specs. The decoder is a series of ConvTranspose1d
    with parametrization chosen to mirror the parameters in the encoder.

    Parameters
    ----------
    SETTINGS: structure
        configuration settings
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
            decoder_layer_specs["layers"].append({"type":  SETTINGS.CONV1dENCODER.activation_fn})
            i -= 1
            
    # For standardized targets, remove last relu 
    if remove_last_activation and decoder_layer_specs["layers"][-1]["type"] ==  SETTINGS.CONV1dENCODER.activation_fn:
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
    
        
    if not (len( SETTINGS.CONV1dENCODER.conv1d_out_channels) == len(SETTINGS.CONV1D.kernel) \
        == len(SETTINGS.CONV1D.stride) == len(SETTINGS.CONV1D.padding)):
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
        SETTINGS.CONV1D.kernel,
        SETTINGS.CONV1D.stride,
        SETTINGS.CONV1D.padding
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
        layers.append({"type":   SETTINGS.CONV1dENCODER.activation_fn})
        in_channels = out_channels
    
    # Check signal shape after stack of conv layers
    all_nr_channels, all_lengths = _compute_conv_output_dim(in_channels, input_length, {"layers": layers})
    
    if  length <=1:
        print("Signal after stack of conv1d has length <=1")
        return None
    
    # Make encoder specs dictionary
    conv1d_layer_specs = {"layers": layers}
    
    return conv1d_layer_specs, all_nr_channels, all_lengths

def add_layer(layer_specs, new_layer):
    """Add a new layer to the structure of layers sepcs.

    Parameters
    ----------
    layer_specs : structure
        example:
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
    new_layer : structure
        example:
        {
            "type": "linear",
            "params": {
                "in_features": in_features,
                "out_features": out_features
            }
        },
    """
    
    # Validate input
    if "layers" not in layer_specs or not isinstance(layer_specs["layers"], list):
        print("layer_specs must contain a 'layers' key with a list of layers.")
        return None

    if not isinstance(new_layer, dict) or "type" not in new_layer:
        print("new_layer must be a dict with at least a 'type' key.")
        return None

    # Append the new layer
    layer_specs["layers"].append(new_layer)
    return layer_specs

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
