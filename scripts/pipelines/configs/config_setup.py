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

        # Direct initialisation (safe because each class handles missing keys internally)
        if "nn_model" in config.keys():
            self.NEURALNET = NNSettings(config)
        if "beta-vae" in config.keys():
            self.BETA_VAE = BetaVae(config)
        if  "conv1d" in config.keys():
            self.CONV1D = Conv1D(config)
            self.CONV1dENCODER = Conv1dEncoderSettings(config)
            
        self.TIME_SEGMENTATION = TimeSettings(config)
        self.TRAINING = TrainingSettings(config)
        self.LOCAL_PATHS = LocalPaths(config)
        self.DATA = DataInput(config)
        self.SCHEDULER = Scheduler(config)


# ======================================================================================================================
class BetaVae:
    def __init__(self, config):
        beta_vae_specs = config.get("beta-vae", {})

        # Assign None if missing, and log warnings
        self.latent_dim = self._get_key(beta_vae_specs, "latent_dim")
        self.beta = self._get_key(beta_vae_specs, "beta")
        self.ref_freq = self._get_key(beta_vae_specs, "ref_freq")
        self.existing_fitted_params = self._get_key(beta_vae_specs, "existing_fitted_params")
        self.hidden_dim = self._get_key(beta_vae_specs, "hidden_dim")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in beta-vae. Setting to None.")
            return None



# ======================================================================================================================
class Conv1dEncoderSettings:
    def __init__(self, config):
        encoder_specs = config.get("encoder_specs", {})

        # Assign None if missing, and log warnings
        self.conv1d_in_channels = self._get_key(encoder_specs, "conv1d_in_channels")
        self.conv1d_out_channels = self._get_key(encoder_specs, "conv1d_out_channels")
        self.activation_fn = self._get_key(encoder_specs, "activation_fn")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in encoder_specs. Setting to None.")
            return None

        
# ======================================================================================================================
class NNSettings:
    def __init__(self, config):
        nn_specs = config.get("nn_model", {})

        self.lr = self._get_key(nn_specs, "lr")
        self.l1_size = self._get_key(nn_specs, "l1_size")
        self.l2_size = self._get_key(nn_specs, "l2_size")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in nn_model configuration. Setting to None.")
            return None


# ======================================================================================================================
class Scheduler:
    def __init__(self, config):
        scheduler = config.get("scheduler", {})
    
        self.mode = self._get_key(scheduler, 'mode') 
        self.factor = self._get_key(scheduler, 'factor')
        self.threshold = self._get_key(scheduler, 'threshold')
        self.threshold_mode =  self._get_key(scheduler, 'threshold_mode')
    
    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in scheduler. Setting to None.")
            return None      
# ======================================================================================================================
class TimeSettings:
    def __init__(self, config):
        time_specs = config.get("time_settings", {})

        # Assign None if missing, and log warnings
        self.time_window_sec = self._get_key(time_specs, "time_window_sec")
        self.stride_sec = self._get_key(time_specs, "stride_sec")
        self.offset = self._get_key(time_specs, "offset")
        self.x_window_sec = self._get_key(time_specs, "x_window_sec")
        self.y_window_sec = self._get_key(time_specs, "y_window_sec")
        self.dt_sec = self._get_key(time_specs, "dt_sec")
        self.stride_unitary = self._get_key(time_specs, "stride_unitary")
        self.targeted_time_stamps_per_window = self._get_key(time_specs, "targeted_time_stamps_per_window")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in time_settings. Setting to None.")
            return None



# ======================================================================================================================
class TrainingSettings:
    def __init__(self, config):
        training_specs = config.get("training", {})

        # Assign None if missing, and log warnings
        self.lr = self._get_key(training_specs, "lr")
        self.num_epochs = self._get_key(training_specs, "num_epochs")
        self.min_nr_epochs = self._get_key(training_specs, "min_nr_epochs")
        self.patience = self._get_key(training_specs, "patience")
        self.slope_threshold = self._get_key(training_specs, "slope_threshold")
        self.weight_decay = self._get_key(training_specs, "weight_decay")
        self.min_increment = self._get_key(training_specs, "min_increment")
        self.dataloader_batch_size = self._get_key(training_specs, "dataloader_batch_size")
        self.train_batch_size = self._get_key(training_specs, "train_batch_size")
        self.min_batch_size = self._get_key(training_specs, "min_batch_size")
        self.num_workers = self._get_key(training_specs, "num_workers")
        self.num_train_samples = self._get_key(training_specs, "num_train_samples")
        self.num_val_samples = self._get_key(training_specs, "num_val_samples")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in training configuration. Setting to None.")
            return None


# ======================================================================================================================
class LocalPaths:
    def __init__(self, config):
        paths_specs = config.get("paths", {})

        self.global_mean_std_path = self._get_key(paths_specs, "global_mean_std_path")
        self.joblib_directory = self._get_key(paths_specs, "joblib_directory")
        self.data_split_csv_path = self._get_key(paths_specs, "data_split_csv_path")
        self.data_output_directory = self._get_key(paths_specs, "data_output_directory")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in paths configuration. Setting to None.")
            return None

# ======================================================================================================================
class DataInput:
    def __init__(self, config):
        local_specs = config.get("local", None)
        input_specs = config.get("input", {})

        self.local = local_specs if local_specs is not None else None
        if self.local is None:
            print("[Warning] Missing 'local' section in configuration. Setting to None.")

        self.data_names = self._get_key(input_specs, "data_names")
        self.target_names = self._get_key(input_specs, "target_names")

        # Compute combined list safely
        if self.data_names and self.target_names:
            self.all_source_signal_list = self.data_names + self.target_names
        else:
            self.all_source_signal_list = None
            print("[Warning] Could not create all_source_signal_list due to missing data_names or target_names.")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in input configuration. Setting to None.")
            return None

# ======================================================================================================================
class Conv1D:
    def __init__(self, config):
        conv_specs = config.get("conv1d", {})

        self.kernel = self._get_key(conv_specs, "kernel")
        self.stride = self._get_key(conv_specs, "stride")
        self.padding = self._get_key(conv_specs, "padding")

    def _get_key(self, section, key):
        if key in section:
            return section[key]
        else:
            print(f"[Warning] Missing key '{key}' in conv1d configuration. Setting to None.")
            return None

    
# ======================================================================================================================
if __name__ == "__main__":
    import json

    config_file_path_ = "scripts/pipelines/configs/config_beta_vae.json"
    settings = get_settings(config_file_path_)

