import os
import json
    
def load_config(config_file_path):
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file {config_file_path} not found.")       
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    return config
    
   
def get_settings(config_file_path):
    config = load_config(config_file_path)
    try:
        return Settings(config)
    except Exception as e:
        print(f"Config validation error: {e}")
        raise


class Settings():
    def __init__(self, config):
        # Set up individual settings classes
        self.PCASETTINGS = PcaSettings(config)
        self.SIGFILL = SigFillSettings(config)
        self.GENERAL = GeneralSettings(config)
        self.LOCALPATHS = PathSettings(config)
       
class PcaSettings():
    def __init__(self, config):
        try:
            self.max_components =  config["pca"]["max_components"]
            self.source_signal_names = config["pca"]["source_signal_names"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
            raise
        
class SigFillSettings():
    def __init__(self, config):
        try:
            self.list_of_signals = config["sigfill"]["list_of_signals"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
            raise


class GeneralSettings():
    def __init__(self, config):
        try:
            self.nr_shots = config["general"]["nr_shots"]
            self.processes = config["general"]["processes"]
            self.local = config["general"]["local"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
            raise
        
class PathSettings():
    def __init__(self, config):
        try:
            self.signal_list_file = config["paths"]["signal list_file"]
            self.output_path = config["paths"]["output_path"]
            self.data_split_file = config["paths"]["data_split_file"]
        except KeyError as e:
            print(f"Missing key in configuration: {e}")
            raise
