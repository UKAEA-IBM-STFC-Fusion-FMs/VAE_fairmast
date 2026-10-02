import os
import json
# ----------------------------------------------------------------------------------------------------------------------

def load_benchmark_config(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    return config

# ----------------------------------------------------------------------------------------------------------------------

def get_benchmark_settings(config_file_path):
    try:
        config = load_benchmark_config(config_file_path)
        return SettingsBenchmark(config)
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
            print(f"Benchmark missing field '{field}'.")
            return None
        target = getattr(self, field)

        # Fetch attribute
        if not hasattr(target, attr):
            print(f"Missing attribute '{attr}' in SettingsBenchmark field '{field}'.")
            return None

        return getattr(target, attr)
    
# ======================================================================================================================

class SettingsBenchmark:
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
        
        if ("signal_model" in config.keys() and "mask_model" in config.keys() and "end_model" in config.keys()):
            self.MODEL = Models(config)
        else:
            raise KeyError("One or more `models` not found in config")
        
        if "local" in config.keys():
            self.local = config["local"]
        else:
            print("'local' not defined in benchmark_setup; settting to True")
            self.local = True
        
        if "output_signals_len" in config.keys():
            self.output_signals_len = config["output_signals_len"]
        else:
            self.output_signals_len = None
            
        if "cache_data" in config.keys():
            self.cache = config["cache_data"]
        else:
            print("'cache_data' not defined in benchmark_setup; setting to False")
            self.cache = False
        
        if "task" not in config.keys():
            raise KeyError("'task' not defined in benchmark_setup, interrupting execution.")

        self.task = config["task"]

        if "fine_tuning" not in config.keys():
            self.fine_tuning = False
        else:
            self.fine_tuning = config["fine_tuning"]

        
# ----------------------------------------------------------------------------------------------------------------------

class TrainingSettings:
    def __init__(self, config):
        training_specs = config.get("training", {})

        # Assign None if missing, and log warnings
        self.lr = self._get_key(training_specs, "lr")
        self.num_epochs = self._get_key(training_specs, "num_epochs")
        
        self.min_nr_epochs = self._get_key(training_specs, "min_nr_epochs")
        if self.min_nr_epochs > self.num_epochs:
            print(f"min_nr_epochs cannot be larger than num_epochs: {self.min_nr_epochs} {self.num_epochs}")
            print("settings min_nr_epochs = {self.num_epochs}")
            self.min_nr_epochs = self.min_nr_epochs 
        
        self.patience = self._get_key(training_specs, "patience")
        if self.patience > self.num_epochs:
            print(f"patience cannot be larger than num_epochs: {self.patience} {self.num_epochs}")
            print("settings patience = {self.num_epochs}")
            self.patience = self.min_nr_epochs 
            
        
        self.dataloader_batch_size = self._get_key(training_specs, "dataloader_batch_size")
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
        self.input_vae_models = self._get_key(paths_specs, "input_vae_models")
        self.actuator_vae_models = self._get_key(paths_specs, "actuator_vae_models")
        self.output_vae_models = self._get_key(paths_specs, "output_vae_models")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in paths configuration. Setting to None.")
            return None

# ----------------------------------------------------------------------------------------------------------------------
class Models:
    def __init__(self, config):

        signal_model = config.get("signal_model", {})
        if signal_model:
            self.signal_layers = self._get_key(signal_model,"layers")
        else:
            self.signal_layers = None
        
        mask_model = config.get("mask_model", {})
        if mask_model:
            self.mask_layers = self._get_key(mask_model,"layers")
        else:
            self.mask_layers = None
            
        end_model = config.get("end_model", {})
        if end_model:
            self.end_layers = self._get_key(end_model, "layers")
        else:
            self.end_layers = None
            
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in model. Setting to None.")
            return None