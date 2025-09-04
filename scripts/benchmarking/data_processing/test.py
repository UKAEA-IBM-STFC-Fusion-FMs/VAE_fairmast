"""PCAnalysis class for performing PCA on MAST data.
"""

import joblib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from config_files.config_setup import get_settings

import sys

cwd = os.path.dirname(os.path.abspath(__file__))
mother_dir = os.path.dirname(cwd)
scripts_dir = os.path.join(os.path.dirname(mother_dir))
sys.path.append(scripts_dir)

from MAST_tools.store_utils import MASTStorageManager
from sigfill import run_simple_filler
from utils import shuffle_shot_ids
from pipelines.utils.utils import read_data_split_csv


def process_single_shot_id( 
    shot_id:int,
    source:str,
    signal_name:str, 
    store_manager: MASTStorageManager,
    local: bool
    ):
    """Process a single shot_id to extract and fill the signal data.

    Parameters
    ----------
    shot_id : int
        shot id to process
    source : str
        name of the source to which the signal belongs
    signal_name : str
        name of the signal to process
    store_manager : MASTStorageManager
        instance of MASTStorageManager to handle data storage
    local : bool
        True if accessing data locally, False if accessing remotely

    Returns
    -------
    np.ndarray
        Processed data for the given shot_id
    """
    
    return run_simple_filler(
        shot_id,
        source,
        signal_name, 
        store_manager,
        local
        )
    
def process_single_shot_id_star(args):
    """Wrapper for process_single_shot_id for multiprocessing starmap."""
    return process_single_shot_id(*args)


class MASTpca:
    """PCAnalysis class for performing PCA on MAST data.
    """
    def __init__(self, SETTINGS):
        self.source_signal_names = SETTINGS.PCASETTINGS.source_signal_names
        self.max_components = SETTINGS.PCASETTINGS.max_components
        self.processes = SETTINGS.GENERAL.processes
        self.nr_shots = SETTINGS.GENERAL.nr_shots
        self.local = SETTINGS.GENERAL.local
        self.output_directory = SETTINGS.LOCALPATHS.output_path
        self.store_manager = MASTStorageManager()
        
    def fit_PCA(
        self,
        shot_ids: list[int],
        source: str,
        signal_name: str,
        ):
        """Aply PCA 

        Parameters
        ----------
        shot_ids : list[int]
            List of shots to use for PCA analysis 
        signal_name : str
            Name of the signal to process
        source : str
            Name of the source to which the signal belongs
        """
        # Use SLURM setting if available
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

        if slurm_cpus is not None:
            self.processes = int(slurm_cpus)
        else:
            # Fallback to user-provided or safe default
            if self.processes is not None and self.processes > 0:
                # Cap to system CPU count
                self.processes = min(self.processes, os.cpu_count() or 1)
            else:
                # Default to 1 if nothing was specified
                self.processes = 1
                

        # Process shot ids
        if self.nr_shots < len(shot_ids):
            shot_ids = shot_ids[:self.nr_shots]
        
        with Pool(processes=self.processes) as pool:
            args_iterable = [
                (shot_id, source, signal_name, self.store_manager, self.local) 
                for shot_id in shot_ids
            ]
            
            filter_none_results_iterator = (
                result for result in pool.imap(process_single_shot_id_star, args_iterable)
                if result is not None
            )
            results = list(tqdm(
                filter_none_results_iterator,
                total=len(shot_ids)
            ))

        if results:
            data_array = np.concatenate([*results], axis=1).T
        else:
            print("Warning: No results to concatenate.")
            return

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_array)

        # Determine the number of components to explain 
        # at least 99.7% of the variance
        for n_components in range(1,self.max_components):
            pca = PCA(n_components)
            pca.fit(scaled_data)

            explained_variance = pca.explained_variance_ratio_

            if np.cumsum(explained_variance)[-1] >= 0.997:
                break
        
        # Save pca model to output_dir 
        os.makedirs(self.output_directory, exist_ok=True)
        
        my_pca_file_name = self.output_directory+f"/pca_{signal_name}.joblib"
        print(f"Saving PCA model to {my_pca_file_name}")
        
        joblib.dump({
            "pca": pca,
            "scaler": scaler,
            "source": source,
            "signal_name": signal_name
             },
            my_pca_file_name) 
        
    def __call__(self, shot_ids):
        for source_signal_names in self.source_signal_names:
            source, signal_name = source_signal_names.split("-")
            print(f"Processing PCA for source: {source}, signal: {signal_name}")
            # Fit PCA for each source and signal name
            self.fit_PCA(shot_ids, source, signal_name)
    

def run_MASTpca(SETTINGS): 
    
    data_split_file = SETTINGS.LOCALPATHS.data_split_file
    
    # Instance of PCAnalysis
    mast_pca = MASTpca(SETTINGS)
    
    # Prepare sample of shot IDs
    shot_ids, _, _ = read_data_split_csv(csv_path = data_split_file)
    shot_ids = shuffle_shot_ids(shot_ids)
    
    # Fit pca to MAST shots
    local = SETTINGS.GENERAL.local
    output_directory = SETTINGS.LOCALPATHS.output_path
    mast_pca(shot_ids)
   
    
def test_reconstruction(
    shot_id, 
    local,
    pca_file_name,
    source,
    signal_name,  
    ):
    
    # Instantiate MASTStorageManager
    store_manager = MASTStorageManager()

    # Demonstrate PCA transform and reconstruction
    models = joblib.load(pca_file_name)
    pca = models["pca"]
    scaler = models["scaler"]

    vals = process_single_shot_id(
        shot_id,
        source,
        signal_name, 
        store_manager,
        local
        )

    if vals is None:
        return
    
    vals_scaled = scaler.transform(vals.T)

    transformed = pca.transform(vals_scaled)
    reconstructed = pca.inverse_transform(transformed)
    reconstructed = scaler.inverse_transform(reconstructed)

    RMS = np.sqrt(np.mean((vals - reconstructed.T) ** 2))
    print(f"RMS = {RMS}")

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    vmin = min(vals.min(), reconstructed.T.min())
    vmax = max(vals.max(), reconstructed.T.max())

    p0 = axes[0].pcolorfast(vals, vmin=vmin, vmax=vmax)
    p1 = axes[1].pcolorfast(reconstructed.T, vmin=vmin, vmax=vmax)


    axes[0].set_title(r"$\bf{Original}$" + f" {source}/{signal_name} shot id = {shot_id}")
    fig.colorbar(p0, ax=axes[0])

    axes[1].set_title(r"$\bf{Reconstructed}$" + f" (RMS = {RMS:.4f})")
    fig.colorbar(p1, ax=axes[1])

    plt.tight_layout()


    fig.savefig(f"{source}_{signal_name}.png", bbox_inches="tight", dpi=300)
    #plt.show() 
 
  
if __name__ == "__main__":
    # Load settings
    SETTINGS = get_settings("scripts/benchmarking/data_processing/config_files/config.json")
    
    # Run PCA analysis
    run_MASTpca(SETTINGS)
    
    
    
    
    
    