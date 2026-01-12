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
from src.pipelines.utils.layer_factory import SequentialBuilder
from src.pipelines.models.conv1d_utils import add_layer



class Encoder():
    def __init__(
        self, 
        SETTINGS):
        
        """_summary_
     
        """

        if SETTINGS.ENCODER_SPECS.encoder_type == "conv1d":
           encoder, decoder, intermediate_layer_size, conv1d_out_dim =  build_conv1d_encoder_decoder(SETTINGS)
            
def conv1d_encoder_specs(SETTINGS):
    
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
        
    nr_channels, length, _, _ = _compute_conv_output_dim(in_channels, input_length, {"layers": layers})

    encoder_layer_specs = {"layers": layers}
    shape = [nr_channels, length]
    return encoder_layer_specs, shape
