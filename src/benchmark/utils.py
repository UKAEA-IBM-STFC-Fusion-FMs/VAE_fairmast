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
    
from tokamark.src.tokamark.tools.utils import get_config_from_yaml
from src.benchmark.configs.benchmark_setup import get_benchmark_settings
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import beta_VAE


def load_task_config(yaml_file_path: str):
    """Load YAML configuration."""
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_file_path}")
    try:
        return get_config_from_yaml(yaml_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config from '{yaml_file_path}': {e}") from e


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
    

    