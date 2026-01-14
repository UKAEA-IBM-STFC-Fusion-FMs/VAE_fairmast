
from .conv1d_utils import build_conv1d_encoder_decoder


class EncoderDecoder():
    def __init__(self,SETTINGS):

        if SETTINGS.ENCODER_SPECS.encoder_type == "conv1d":
            results = build_conv1d_encoder_decoder(SETTINGS)
            if results is None:
                self.encoder = self.decoder = self.intermediate_layer_size = self.conv1d_out_dim = None
            else:
                self.encoder, self.decoder, self.intermediate_layer_size, self.conv1d_out_dim = results