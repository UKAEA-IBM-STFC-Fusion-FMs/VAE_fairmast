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

    if signal_name == "flux_loop_flux":
        encoder_layer_specs = {
                "layers": [
                    {
                        "type": "conv1d",
                        "params": {
                            "in_channels": 15,
                            "out_channels": 40,
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
                            "in_channels": 40,
                            "out_channels": 20,
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
                            "in_channels": 20,
                            "out_channels": 20,
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
    
def FullyConnectedEncode(in_features, out_features):
    layer_specs = {
            "layers": [
                {
                    "type": "linear",
                    "params": {
                        "in_features": in_features,
                        "out_features": int(out_features*2)
                    }
                },
                {
                    "type": "relu"
                },
                {
                    "type": "linear",
                    "params": {
                        "in_features": int(out_features*2),
                        "out_features": out_features
                    }
                },
                {
                    "type": "relu"
                }
            ]
        }
    return layer_specs