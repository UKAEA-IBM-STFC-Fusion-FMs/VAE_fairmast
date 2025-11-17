import math
import torch
import torch.nn as nn


def build_conv1d_encoder_decoder(SETTINGS, input_channels, input_length):
    """Build encoder and decoder specs.
    The encoder architecture can be writte manually or by using the _build_encoder_layer_specs
    which reads it directly from the config.json file.

    Parameters
    ----------
    SETTINGS : dict
        settings from config.json.
    input_channels : int
        Nr of channels in the input signal.
    input_length : int
        Length along the time dimension of the input signal.

    Returns
    -------
    dictionaries
        encoder specs as dictionary, 
        encoded signal shape [int, int],
        decoder specs as dictionary.

    Raises
    ------
    ValueError
        If signal after encoder has length <=1
    """
    
    # Build conv1d encoder
    conv1d_encoder_specs, encoded_signal_shape = _build_encoder_layer_specs(SETTINGS, input_length)
    print(f"ENCODER SPECS {conv1d_encoder_specs}")
    
    last_nr_channels, last_signal_length, _, _ = _compute_conv_output_dim(input_channels, input_length, conv1d_encoder_specs)
    encoded_signal_shape = [last_nr_channels, last_signal_length]
    
    if  last_signal_length <=1:
        raise ValueError("Signal after stack of conv1d has length <=1")

    conv1d_decoder_specs = _build_decoder_specs_from_encoder_specs(SETTINGS, conv1d_encoder_specs, input_channels, input_length, remove_last_activation = True)
    
    # Uncomment this to hardcode conv1d_encoder_specs
    # Template:
    # conv1d_encoder_specs = {
    #         "layers": [
    #             {
    #                 "type": "conv1d",
    #                 "params": {
    #                     "in_channels": 15,
    #                     "out_channels": 128,
    #                     "kernel_size": 4,
    #                     "stride": 1,
    #                     "padding": SETTINGS.CONV1D.padding
    #                 }
    #             },
    #             {
    #                 "type": "relu"
    #             }
    #         ]
    #     }
    
    return conv1d_encoder_specs, encoded_signal_shape, conv1d_decoder_specs
    
    
def _build_encoder_layer_specs(SETTINGS, input_length):
    """Dynamically builds the encoder layers specs based on the SETTINGS:

    Parameters
    ----------
    SETTINGS : 
        Dictionary from json file.
    input_length: int 
        length of the input signal.

    Returns
    -------
    list of layers spec
    """
    
    
    if not (len( SETTINGS.ENCODER.conv1d_out_channels) == len(SETTINGS.CONV1D.kernel) \
        == len(SETTINGS.CONV1D.stride) == len(SETTINGS.CONV1D.padding)):
        raise ValueError(
            f"Mismatch in layer specs: "
            f"out_channels={len(out_channels_list)}, "
            f"kernel={len(kernel_list)}, "
            f"stride={len(stride_list)}, "
            f"padding={len(padding_list)}"
        )

    in_channels = SETTINGS.ENCODER.conv1d_in_channels
    
    layers = []
    for out_channels, kernel, stride, padding in zip(
        SETTINGS.ENCODER.conv1d_out_channels,
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
        layers.append({"type":   SETTINGS.ENCODER.activation_fn})
        in_channels = out_channels
        
    nr_channels, length, _, _ = _compute_conv_output_dim(in_channels, input_length, {"layers": layers})

    encoder_layer_specs = {"layers": layers}
    shape = [nr_channels, length]
    return encoder_layer_specs, shape

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

    return current_channels, current_length, lengths, channels

def _convt1d_needed_output_padding(L_in, L_out_target, k, s=1, p=0, d=1):
    # L_out_target = desired output of convtranspose
    # L_in = input length to convtranspose (which equals encoder's L_out at that stage)
    base = (L_in - 1) * s - 2*p + d*(k-1) + 1
    op = L_out_target - base
    return op

def _build_decoder_layers_from_encoder_specs(encoder_layer_specs, input_channels, input_length):
    """Define decoder layers for the conv1d encoder. These are a series of ConvTranspose1d.

    Parameters
    ----------
    encoder_layer_specs : list[dict]
        dictionary specifying encoder layers.
    input_channels : int
        original number of channels
    input_length : int 
        original input length (time dimension).   

    Returns
    -------
    list[nn.ConvTranspose1d]
    """
    
    # Pass 1: compute encoder output dimensions
    _,_,enc_len,enc_c = _compute_conv_output_dim(input_channels, input_length, encoder_layer_specs)
    # enc_len[0] original input length, enc_len[i] output length from i-th conv1d layer
    # enc_c[0] original nr of input channels, enc_c[i] output nr. channels from i-th conv1d layer
    print(f"encoded layers length {enc_len}")
    # Pass 2: build decoder layers in reverse
    dec_layers = []

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
            layer = nn.ConvTranspose1d(
                in_channels=enc_c[i],
                out_channels=enc_c[i-1],   # mirror channels
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                output_padding=op
            )
            dec_layers.append(layer)
            i -= 1
           
    return dec_layers

def _build_decoder_specs_from_encoder_specs(
    SETTINGS,
    encoder_layer_specs, 
    input_channels, 
    input_length,
    remove_last_activation = True
    ):
    """
    encoder_layer_specs: dictionary specifying encoder layers.
    input_channels: original number of channels.
    input_length: original input time length (L0).
    """
    dec_layers = _build_decoder_layers_from_encoder_specs(
        encoder_layer_specs, 
        input_channels, 
        input_length
        )
    decoder_layer_specs = {"layers": []}
    
    for layer in dec_layers:
        if isinstance(layer, nn.ConvTranspose1d):
            # Add ConvTranspose1d spec
            spec = {
                "type": "conv_transpose1d",
                "params": {
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size[0], 
                    "stride": layer.stride[0],
                    "padding": layer.padding[0],
                    "dilation": layer.dilation[0],
                    "output_padding": layer.output_padding[0],
                }
            }
            decoder_layer_specs["layers"].append(spec)
            
            # Add ReLU spec after each ConvTranspose1d
            decoder_layer_specs["layers"].append({"type":  SETTINGS.ENCODER.activation_fn})
    
    # For standardized targets, remove last relu 
    if remove_last_activation and decoder_layer_specs["layers"][-1]["type"] ==  SETTINGS.ENCODER.activation_fn:
        decoder_layer_specs["layers"].pop()

    return decoder_layer_specs

def FullyConnectedEncode(in_features, out_features):
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
