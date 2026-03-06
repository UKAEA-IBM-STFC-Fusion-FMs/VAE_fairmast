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
    
# ----------------------------------------------------------------------------------------------------------------------

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

class Settings:
    def __init__(self, config):
        self.config = config

        # Check presence of essential attributes in config
        if "paths" in config.keys():
            self.LOCAL_PATHS = LocalPaths(config)
        else:
            raise KeyError("'paths' not found in config")
        
        if "training" in config.keys():
            self.TRAINING = TrainingSettings(config)
        else:
            raise KeyError("'training' not found in config")
        
        if "local" not in config:
            raise KeyError("'local' not defined in benchmark_setup; defaulting to True")

        self.local = config.get("local", True)

        if "cache_data" not in config:
            raise KeyError("'cache_data' not defined in benchmark_setup; defaulting to True")

        self.cache = config.get("cache_data", True)
        
        if "task" not in config:
            raise KeyError("'task' not defined in benchmark_setup; defaulting to True")

        self.task = config.get("task", None)

        
# ----------------------------------------------------------------------------------------------------------------------

class TrainingSettings:
    def __init__(self, config):
        training_specs = config.get("training", {})

        # Assign None if missing, and log warnings
        self.lr = self._get_key(training_specs, "lr")
        self.num_epochs = self._get_key(training_specs, "num_epochs")
        self.min_nr_epochs = self._get_key(training_specs, "min_nr_epochs")
        self.dataloader_batch_size = self._get_key(training_specs, "dataloader_batch_size")
        self.train_batch_size = self._get_key(training_specs, "train_batch_size")
        self.num_workers = self._get_key(training_specs, "num_workers")
        self.num_train_samples = self._get_key(training_specs, "num_train_samples")
        self.num_val_samples = self._get_key(training_specs, "num_val_samples")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in training configuration. Setting to None.")
            return None
        
# ----------------------------------------------------------------------------------------------------------------------

class LocalPaths:
    def __init__(self, config):
        paths_specs = config.get("paths", {})

        self.global_mean_std_path = self._get_key(paths_specs, "global_mean_std_path")
        self.data_split_csv_path = self._get_key(paths_specs, "data_split_csv_path")
        self.vae_directory = self._get_key(paths_specs, "vae_directory")
        self.output_directory = self._get_key(paths_specs, "output_directory")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in paths configuration. Setting to None.")
            return None

# ----------------------------------------------------------------------------------------------------------------------
