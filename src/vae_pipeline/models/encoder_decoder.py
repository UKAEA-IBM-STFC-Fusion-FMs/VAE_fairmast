from .encoder_decoder_utils import (build_conv1d_encoder_decoder, 
                                    build_linear_encoder_decoder,
                                    quick_build_from_config
                                    )


class EncoderDecoder():
    def __init__(self,SETTINGS):

        if SETTINGS.ENCODER.type == "conv1d":
            results = build_conv1d_encoder_decoder(SETTINGS)
            
        if SETTINGS.ENCODER.type == "linear":
            results = build_linear_encoder_decoder(SETTINGS)
        
        if SETTINGS.ENCODER.type == "encoder_decoder":
            results = quick_build_from_config(SETTINGS)
            
        if results is None:
            self.encoder = self.decoder = self.size_before_vae  = None
        else:
            self.encoder, self.decoder, self.size_before_vae = results
