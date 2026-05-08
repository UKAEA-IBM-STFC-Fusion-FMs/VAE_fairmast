import argparse
import os
import sys


REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
from src.benchmark.configs.benchmark_setup import get_benchmark_settings
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE

def load_benchmark_settings(config_file_path: str):
    """Load JSON settings."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"JSON configuration file not found: {config_file_path}")
    try:
        return get_benchmark_settings(config_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON config from '{config_file_path}': {e}") from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load task YAML and pipeline JSON configuration."
    )
    parser.add_argument(
        "--config_task_file_path",
        default="",
        type=str,
        help="Path to configuration YAML task file."
    )
    parser.add_argument(
        "--config_benchmark_file_path",
        default="src/benchmark/configs/task1_1_config.json",
        type=str,
        help="Path to configuration JSON file for the pipeline."
    )
    return parser.parse_args()


def load_vae_model(config_path:str):
    """
    config_path : str
        Path to the config.json containing parameters for initializing the model
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.") 
    else:
        try:
            settings = get_settings(config_path) 
        except Exception as e:
            print(f"Error in loading configuration {e}")
            return None
    
    try:
        model = beta_VAE(settings)
    except Exception as e:
        print(f"Error in initializing vae model: {e}")
        return None
            
    return model
    

def create_vae_dictionary(vae_dictionary, type_of_signal, list_of_signals, SETTINGS, model_sub_paths):
    """ Loads VAE models into a dictionary for signals in the list.

       Reinforce the one-to-one correspondence between input or actuator signals and VAE models.

    Parameters
    ----------
    vae_dictionary : _type_
       vae_dictionary = {"input": {}, "actuator": {}, "output": {}}
    type_of_signal : str
        either "input" or "actuator" or "output"
    list_of_signals : list[str]
        List of signals from task config file
    SETTINGS : SettingsBenchmark
        Settings from json file
    model_sub_paths : list[str]
        paths to models

    Return
    --------
        Filled vae_dictionary
    """
    if not list_of_signals:
        return vae_dictionary

    if type_of_signal not in [ "input","actuator", "output"]:
        raise ValueError(f"Type of signal specified {type_of_signal} not in the list of correct keys:  [input, actuator,output] ")

    for i, (source, signal_name) in enumerate(list_of_signals):
        key = f"{source}-{signal_name}"
        vae_dictionary[type_of_signal][key] = None

        for model_path in model_sub_paths:
            if signal_name in model_path:
                
                full_path = os.path.join(SETTINGS.LOCAL_PATHS.vae_directory, model_path)

                # store model keyed by source-signal
                vae_dictionary[type_of_signal][key] = load_vae_model(full_path)
                break

    num_inputs = len(list_of_signals)
    num_loaded = len(vae_dictionary[type_of_signal])

    if num_inputs != num_loaded:
        if type_of_signal in ["input", "actuator"]:
            raise ValueError(
                f"The number of input signals ({num_inputs}) does not match the number of VAE "
                f"models loaded ({num_loaded})."
            )

    return vae_dictionary
    