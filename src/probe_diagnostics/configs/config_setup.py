import os
import json


# ----------------------------------------------------------------------------------------------------------------------
def load_config(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    return config


# ----------------------------------------------------------------------------------------------------------------------
def get_settings(config_file_path):
    try:
        config = load_config(config_file_path)
        return Settings(config)
    except Exception as e:
        print(f"Config validation error: {e}")
        raise


# ======================================================================================================================
class Settings:
    def __init__(self, config):
        self.config = config

        # Check presence of essential attributes in config
        if "beta-vae" in config.keys():
            self.BETA_VAE = BetaVae(config)
        else:
            raise KeyError("'beta-vae' not found in config")
        
        if "input" in config.keys():
            self.DATA = DataInput(config)
        else:
            raise KeyError("'input' not found in config")
        
        if "paths" in config.keys():
            self.LOCAL_PATHS = LocalPaths(config)
        else:
            raise KeyError("'paths' not found in config")
        
        if "time_settings" in config.keys():
            self.TIME_SEGMENTATION = TimeSettings(config)
        else:
            raise KeyError("'time_settings' not found in config")
        
        if "windowed_data_specs" in config.keys():
            self.WINDOWsSHAPE = WindowShape(config)
        else:
            raise KeyError("'windowed_data_specs' not found in config")

        if "training" in config.keys():
            self.TRAINING = TrainingSettings(config)
        else:
            raise KeyError("'training' not found in config")
        
        if "encoder" in config.keys():
            self.ENCODER = Encoder(config)
        else:
            raise KeyError("'encoder' not found in config")
        
        if "decoder" in config.keys():
            self.DECODER = Decoder(config)
        
    def get(self, field:str, attr:str):
        """Return attribute `attr` from `field` if it exists; otherwise None.

        Parameters
        ----------
        field : str
            Name of a Settings attribute (e.g., 'ENCODER_SPECS') 
        attr : str
            The name of the attribute to fetch from `field`.

        Returns
        -------
        Any or None
            The value of `field.attr`, or None if missing. Prints a helpful message on failure.
        """
        # Resolve field if a string is passed
        
        if not hasattr(self, field):
            print(f"Settings missing field '{field}'.")
            return None
        target = getattr(self, field)

        # Fetch attribute
        if not hasattr(target, attr):
            print(f"Missing attribute '{attr}' in Settings field '{field}'.")
            return None

        return getattr(target, attr)

# ======================================================================================================================
class Encoder:
    def __init__(self, config):
        encoder = config.get("encoder", {})
        self.layers = self._get_key(encoder,"layers")
        self.type = self._get_key(encoder,"type")
        self.activation_fn = self._get_key(encoder,"activation_fn")
        
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in encoder. Setting to None.")
            return None
           
# ======================================================================================================================
class Decoder:
    def __init__(self, config):
        decoder = config.get("decoder", {})
        self.layers = self._get_key(decoder,"layers")
        self.type = self._get_key(decoder,"type")
        self.activation_fn = self._get_key(decoder,"activation_fn")
        
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in decoder. Setting to None.")
            return None
           
# ======================================================================================================================
class BetaVae:
    def __init__(self, config):
        beta_vae_specs = config.get("beta-vae", {})

        # Assign None if missing, and log warnings
        self.latent_dim = int(self._get_key(beta_vae_specs, "latent_dim"))
        self.beta = self._get_key(beta_vae_specs, "beta")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in beta-vae. Setting to None.")
            return None

# ======================================================================================================================
class TimeSettings:
    def __init__(self, config):
        time_specs = config.get("time_settings", {})

        # Assign None if missing, and log warnings
        self.stride_sec = self._get_key(time_specs, "stride_sec")
        self.x_window_sec = self._get_key(time_specs, "x_window_sec")
        self.targeted_time_stamps_per_window = self._get_key(time_specs, "targeted_time_stamps_per_window")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in time_settings. Setting to None.")
            return None
# ======================================================================================================================
class WindowShape:
    def __init__(self, config):
        window_specs = config.get("windowed_data_specs",{})
    
        self.window_channels = self._get_key(window_specs, "window_channels")
        self.window_length = self._get_key(window_specs,"window_length")
    
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in window_data_specs configuration. Setting to None.")
            return None   

# ======================================================================================================================
class LocalPaths:
    def __init__(self, config):
        paths_specs = config.get("paths", {})

        self.global_mean_std_path = self._get_key(paths_specs, "global_mean_std_path")
        self.data_split_csv_path = self._get_key(paths_specs, "data_split_csv_path")
        self.data_split_csv_path = self._get_key(paths_specs, "data_split_csv_path")
        self.data_output_directory = self._get_key(paths_specs, "data_output_directory")
        self.model_path = self._get_key(paths_specs, "model_path")
        
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in paths configuration. Setting to None.")
            return None

# ======================================================================================================================
class DataInput:
    def __init__(self, config):
        self.local = config.get("local", True)

        self.cache_data =  config.get("cache_data", True)
        

        if self.local is None:
            print("[Warning] Missing 'local' section in configuration. Setting to None.")

        input_specs = config.get("input", {})
        self.data_names = self._get_key(input_specs, "data_names")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in input configuration. Setting to None.")
            return None
# ======================================================================================================================      
class TrainingSettings:
    def __init__(self, config):
        training_specs = config.get("training", {})

        # Assign None if missing, and log warnings
        self.dataloader_batch_size = self._get_key(training_specs, "dataloader_batch_size")
        self.num_workers = self._get_key(training_specs, "num_workers")
        self.num_train_samples = self._get_key(training_specs, "num_train_samples")
        self.num_val_samples = self._get_key(training_specs, "num_val_samples")
        self.num_test_samples = self._get_key(training_specs, "num_test_samples")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in training configuration. Setting to None.")
            return None


