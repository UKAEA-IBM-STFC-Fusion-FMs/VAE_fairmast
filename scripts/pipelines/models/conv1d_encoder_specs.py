def encoder_specs(
    SETTINGS,
    signal_name
    ):
    
    if signal_name == "i_plasma":
        encoder_layer_specs = {
            "layers": [
                {
                    "type": "conv1d",
                    "params": {
                        "in_channels": 1,
                        "out_channels": 4,
                        "kernel_size": SETTINGS.CONV1D.kernel,
                        "stride": SETTINGS.CONV1D.stride,
                        "padding": SETTINGS.CONV1D.padding
                    }
                },
                {
                    "type": "relu"
                },
                {
                    "type": "conv1d",
                    "params": {
                        "in_channels": 4,
                        "out_channels": 4,
                        "kernel_size": SETTINGS.CONV1D.kernel,
                        "stride": SETTINGS.CONV1D.stride,
                        "padding": SETTINGS.CONV1D.padding
                    }
                },
                {
                    "type": "relu"
                },
                {
                    "type": "conv1d",
                    "params": {
                        "in_channels": 4,
                        "out_channels": 4,
                        "kernel_size": SETTINGS.CONV1D.kernel,
                        "stride": SETTINGS.CONV1D.stride,
                        "padding": SETTINGS.CONV1D.padding
                    }
                },
                {
                    "type": "relu"
                }  
            ]
        }
    
    return encoder_layer_specs
    
    