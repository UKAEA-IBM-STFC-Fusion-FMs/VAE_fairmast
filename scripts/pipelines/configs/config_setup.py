import json
import os


# ----------------------------------------------------------------------------------------------------------------------
def load_config(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    return config


# ----------------------------------------------------------------------------------------------------------------------
def get_settings(config_file_path):
    config = load_config(config_file_path)
    try:
        return Settings(config)
    except Exception as e:
        print(f"Config validation error: {e}")
        raise


# ======================================================================================================================
class Settings:
    def __init__(self, config):
        # Set up individual settings classes
        self.config = config
        self.NEURALNET = NNSettings(config)
        self.TIME_SEGMENTATION = TimeSettings(config)
        self.TRAINING = TrainingSettings(config)
        self.LOCAL_PATHS = LocalPaths(config)
        self.DATA = DataInput(config)
        self.BETA_VAE = BetaVae(config)


# ======================================================================================================================
class BetaVae:
    def __init__(self, config):
        try:
            # Model parameters
            self.latent_dim = config["beta-vae"]["latent_dim"]
            self.beta = config["beta-vae"]["beta"]
            self.lr = config["beta-vae"]["lr"]
            self.ref_freq = config["beta-vae"]["ref_freq"]
            self.existing_fitted_params = config["beta-vae"]["existing_fitted_params"]
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
class NNSettings:
    def __init__(self, config):
        try:
            # Model parameters
            self.lr = config["nn_model"]["lr"]
            self.l1_size = config["nn_model"]["l1_size"]
            self.l2_size = config["nn_model"]["l2_size"]
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
class TimeSettings:
    def __init__(self, config):
        try:
            # Time settings for data segmentation

            # This is the length of the time window in seconds. for segmentation of data signals.
            self.time_window_sec = config["time_settings"]["time_window_sec"]
            
            # # This is the time step in seconds. Time window is moved by this amount backwards in time.
            self.time_step = config["time_settings"]["time_step"]

            # This is the target offset in seconds. Signal is predicted in this time window.
            self.offset = config["time_settings"]["offset"]
            
            self.x_window_sec = config["time_settings"]["x_window_sec"]
            self.y_window_sec = config["time_settings"]["y_window_sec"]
            self.dt_sec = config["time_settings"]["dt_sec"]
            self.stride_sec = config["time_settings"]["stride_sec"]
            self.stride_unitary = config["time_settings"]["stride_unitary"]
            self.min_samples_per_window = config["time_settings"]["min_samples_per_window"]
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
class TrainingSettings:
    def __init__(self, config):

        # Training parameters
        try:
            self.num_epochs = config["training"]["num_epochs"]
            self.dataloader_batch_size = config["training"]["dataloader_batch_size"]
            self.train_batch_size = config["training"]["training_batch_size"]
            self.min_batch_size = config["training"]["min_batch_size"]
            self.num_workers = config["training"]["num_workers"]
            self.num_train_samples = config["training"]["num_train_samples"]
            self.num_val_samples = config["training"]["num_val_samples"]
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
class LocalPaths:
    def __init__(self, config):
        try:
            # Local paths
            self.average_values_file_path = config["paths"]["average_values_file_path"]
            self.joblib_directory = config["paths"]["joblib_directory"]
            self.data_split_csv_path = config["paths"]["data_split_csv_path"]
            self.data_output_directory = config["paths"]["data_output_directory"]
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
class DataInput:
    def __init__(self, config):
        try:
            # Data lists
            self.local = config["local"]
            self.data_names = config["input"]["data_names"]
            self.target_names = config["input"]["target_names"]
            self.all_source_signal_list = self.data_names + self.target_names
        except KeyError as e:
            print(f"Missing key in training configuration: {e}")
            raise


# ======================================================================================================================
if __name__ == "__main__":
    import json

    config_file_path_ = "scripts/main_pipeline/configs/config_lr_0_0001.json"
    settings = get_settings(config_file_path_)
    print(settings.NEURALNET.lr)
