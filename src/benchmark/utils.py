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
    
from fairmast_data_processing.src.MAST_benchmark.tools.utils import get_config_from_yaml
from src.benchmark.configs.benchmark_setup import get_settings

def load_task_config(yaml_file_path: str):
    """Load YAML configuration."""
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_file_path}")
    try:
        return get_config_from_yaml(yaml_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config from '{yaml_file_path}': {e}") from e


def load_model_settings(config_file_path: str):
    """Load JSON settings."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"JSON configuration file not found: {config_file_path}")
    try:
        return get_settings(config_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON config from '{config_file_path}': {e}") from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load task YAML and pipeline JSON configuration."
    )
    parser.add_argument(
        "--config_task_file_path",
        default="fairmast_data_processing/src/MAST_benchmark/tasks_configs/group_1_reconstruction/task_1-1.yaml",
        type=str,
        help="Path to configuration YAML task file."
    )
    parser.add_argument(
        "--config_model_file_path",
        default="src/benchmark/configs/task1_1_config.json",
        type=str,
        help="Path to configuration JSON file for the pipeline."
    )
    return parser.parse_args()
