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


class Encoder():
    def __init__(
        self, 
        encoder_type,
        encoder_specs, 
        encoded_signal_shape,
        latent_dim):
        
        
        if encoder_type == "conv1d":
            
            conv1d_encoder = SequentialBuilder(encoder_specs)
                    
            # Find shape after encoding
            conv_out_channels = encoded_signal_shape[0]
            conv_out_length  = encoded_signal_shape[1]
            conv_out_dim = encoded_signal_shape[0]*encoded_signal_shape[1]

            # Add Fully Connected Layer to encoder
            FCLlayer_size = int((conv_out_dim + latent_dim) /2)
            FCLencoder = SequentialBuilder(
                FullyConnectedLinearRelu( conv_out_dim, FCLlayer_size )
                )
            
            # # =============== VAE =====================
            # fc_mu = nn.Linear(FCLlayer_size, latent_dim)
            # fc_logvar = nn.Linear(FCLlayer_size, latent_dim)

            # # =============== Decoder =====================
            # FCLdecoder = SequentialBuilder(
            #     FullyConnectedLinearRelu(latent_dim, FCLlayer_size)
            #     )
            # FCLdecoder2 = SequentialBuilder(
            #     FullyConnectedLinearRelu(FCLlayer_size, conv_out_dim)
            #     )

            # conv1d_decoder = SequentialBuilder(conv1d_decoder_layer_specs)