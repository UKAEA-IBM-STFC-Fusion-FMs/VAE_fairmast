import json
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from scripts.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms
)

from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)

from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.imputer_transform import ImputerTransform

from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.models.conv1d_vae_model import Conv1dVAE, loss_function
from scripts.pipelines.models.conv1d_encoder_decoder_specs import build_conv1d_encoder_decoder
from scripts.pipelines.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from scripts.pipelines.collate_functions.collate_functions import Conv1dVAECollate_v2 as Conv1dVAECollate

# Determine device to train on
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--------------- RUNNING ON GPUs ---------------")
else:
    device = torch.device("cpu")
    print(f"--------------- RUNNING ON CPUs ---------------")
    

def fit_mean_and_std_for_signal_transform( 
                                          train_shots,
                                          output_dir, 
                                          source_signal_list,
                                          verbose=False,
                                          use_existing=False, 
                                          local=True
                                          ):
    """
    Fit or load mean and std for signal transformation.

    Args:
        output_sub_dir: Directory to save/load fitted parameters
        verbose: Print verbose output
        use_existing: If True, try to load existing fitted parameters instead of re-fitting
    """
    os.makedirs(output_dir, exist_ok=True)

    mean_path = os.path.join(output_dir, "dict_mean_shot.pkl")
    std_path = os.path.join(output_dir, "dict_std_shot.pkl")

    # Try to load existing files if requested
    if use_existing and os.path.exists(mean_path) and os.path.exists(std_path):
        if verbose:
            print("\n\n----------LOADING EXISTING FITTED PARAMETERS----------\n")
            print(f"Loading fitted parameters from: {output_dir}")

        try:
            with open(mean_path, "rb") as f:
                dict_mean_ = pickle.load(f)
            with open(std_path, "rb") as f:
                dict_std_ = pickle.load(f)

            if verbose:
                print(f"Successfully loaded mean and std dictionaries")
                print(f"Mean dict keys: {list(dict_mean_.keys())}")
                print(f"Std dict keys: {list(dict_std_.keys())}")

            return dict_mean_, dict_std_

        except Exception as e:
            if verbose:
                print(f"Error loading existing fitted parameters: {e}")
                print("Falling back to re-fitting")

    if verbose:
        print("\n\n----------TRANSFORM FITTING----------\n")

    preprocessing_train_dataset = MastDataset(
        local=local,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform=None,
    )

    if verbose:
        print(f"len(preprocessing_train_dataset): {len(preprocessing_train_dataset)}")

    dict_mean_ = get_mean_shot(preprocessing_train_dataset)
    dict_std_ = get_std_shot(preprocessing_train_dataset)

    # Save fitted parameters
    if verbose:
        print(f"Output folder to save fitted mean and std dicts: {output_dir}")

    with open(mean_path, "wb") as f:
        pickle.dump(dict_mean_, f)
    with open(std_path, "wb") as f:
        pickle.dump(dict_std_, f)

    return dict_mean_, dict_std_

def get_train_test_val_shots(
    max_index_for_train=None,
    max_index_for_val = None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index_for_train:
        train_set = train_sh[0:max_index_for_train]
        val_set = val_sh[0:max_index_for_val]

    return train_set, val_set

def initialize_datasets(
        sources_and_signals, 
        shots, 
        signal_transform_map, 
        shot_transforms, 
        local_flag=False
    ):
    
    datasets_ = {"train": None, "val": None}
    data_set_types = ["train", "val"]
    
    for data_set_type in data_set_types:
        if shots[data_set_type]:
            datasets_[data_set_type] = MastDataset(
                local=local_flag,
                shots_list=shots[data_set_type],
                source_signal_list=sources_and_signals,
                signal_level_transform_map=signal_transform_map,
                shot_level_transform=shot_transforms,
            )
    datasets_["train"] = CachedDataset(datasets_["train"])
    datasets_["val"]   = CachedDataset(datasets_["val"])    
    return datasets_

def initialize_dataloaders(
        datasets,
        collate_function,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=False,
        persistent_workers=True
    ):
    
    dataloaders_ = {"train": None, "val": None}

    data_set_types = ["train", "val"]
    
    for data_set_type in data_set_types:
        if datasets[data_set_type]:
            dataloaders_[data_set_type] = DataLoader(
                dataset=datasets[data_set_type],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=collate_function,
            )

    return dataloaders_

def create_conv1d_vae_model(
    SETTINGS,
    dataloader, 
    verbose = False
    ):
    """Create conv1d-VAE model"""
    
    # Get one sample from the batch to determine signal shape 
    sample_batch = next(iter(dataloader))
    
    for group_idx, signal_data in sample_batch.items():

        input_length = signal_data.shape[-1]  # Last dimension is time
        input_channels = signal_data.shape[-2] # Nr. of channels
                
        vae_specs = {
            "beta": SETTINGS.BETA_VAE.beta, 
            "latent_dim": SETTINGS.BETA_VAE.latent_dim, 
            "input_length": input_length
        }

        # Encoder layer specs
        try:
            conv1d_encoder_layer_specs, encoded_signal_shape, conv1d_decoder_layer_specs = build_conv1d_encoder_decoder(
                SETTINGS, 
                input_channels, 
                input_length
            )
        except ValueError as e:
            print(f"Building encoder error: {e}")
            return models

        model = Conv1dVAE(
            conv1d_encoder_layer_specs, 
            encoded_signal_shape,
            conv1d_decoder_layer_specs, 
            vae_specs
            )
        
        break
    
    return model

def plot_histograms(
    properties,
    Nbins,
    color,
    x_label,
    y_label,
    title_prefix,
    file_name,
    num_rows=3,
    num_cols=5,
    x_max = None,
    x_min = None
):
    num_features = len(properties)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))
    axes = axes.flatten()  # Flatten to 1D for easy iteration

    for i in range(num_features):
        ax = axes[i]
        if x_min is not None and x_max is not None:
                ax.hist(properties[i], bins=Nbins, color=color, alpha=0.7, range=(x_min, x_max))
                ax.set_xlim(x_min, x_max)
        else:
            ax.hist(properties[i], bins=Nbins, color=color, alpha=0.7)
        # ax.set_title(f"{title_prefix} {i+1}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend([f"Ch. {i+1}: {len(properties[i])} items"])
    
    # Hide unused subplots if grid > num_features
    for j in range(num_features, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
 
def test_model(source:str, signal_name:str, output_dir:str, SETTINGS):
    """Test pre-trained model 

    Parameters
    ----------
    source : str
        source name
    signal_name : str
        signal name
    output_dir : str
        path to output directory
    SETTINGS : structure
        config settings
    """
    model_path = os.path.join(output_dir, "best_conv1d_vae_" + signal_name + ".pt")
     
    if not os.path.exists(model_path):
        print(f"{model_path} not found")
        return
    else:
        output_directory = os.path.dirname(model_path)

    # HPC settings for CPUs only
    num_workers = SETTINGS.TRAINING.num_workers

    source_signal_list = SETTINGS.DATA.data_names

    # Parameters for window segmentation (no x/y split for VAE)
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,  # 100ms windows
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "tergeted_time_stamp_per_window": SETTINGS.TIME_SEGMENTATION.tergeted_time_stamp_per_window,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, val_shots = get_train_test_val_shots(
        SETTINGS.TRAINING.num_val_samples
    )

    this_signal = "flux_loop_flux"
    # dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
    #     train_shots,
    #     output_directory,
    #     source_signal_list,
    #     verbose=False,
    #     use_existing=SETTINGS.BETA_VAE.existing_fitted_params,
    #     local = SETTINGS.DATA.local
    # )
    
    # Get mean and std for signal transformation
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
        dict_mean = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
        dict_std = pickle.load(f)
        
    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_mean[var], dict_std[var]),
                ImputerTransform(),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transforms = ComposeTransforms(
        [
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            Conv1dVAETransform(),
        ]
    )

 
    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local
    )
    

    conv1d_vae_collate_fn = Conv1dVAECollate(SETTINGS.TRAINING.train_batch_size)
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    val_dataloader = dataloaders_train_val_test["val"]
    
    # Create conv1d-VAE models
    model = create_conv1d_vae_model(
        SETTINGS,
        val_dataloader, 
        verbose = False
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    loss_vs_batch = []
    correlations_ = []
    rel_errors = []
    rmse = []
    best_rel_error = None
    worst_rel_error =None
    x_best_input = None
    x_best_recon = None
    x_worst_input = None
    x_worst_recon = None
    x_input_N = []
    x_recon_N = []
    N=10 # Tensors to plot
    max_error = float("-inf")
    best_loss = float("inf")
    minimum_error = float("inf")

    
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(val_dataloader):
            
            print(f"Batch idx {batch_idx}")
            
            # if batch_idx==1:
            #     break
                
            for group_idx, stacked_tensor in batch.items():
                
                # if group_idx ==1:
                #     break
                
                x = stacked_tensor.to(device)
                x_recon, mu, logvar = model(x)

                # Compute loss
                total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)
                loss_vs_batch.append(total_loss.item())  
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    
                num_channels = x.shape[1]
                if not correlations_:
                    correlations_ = [[] for _ in range(num_channels)]
                if not rel_errors:
                    rel_errors = [[] for _ in range(num_channels)]

                # Compute correlations
                correl = correlations(x, x_recon)
                if isinstance(correl, torch.Tensor):
                    correl = correl.cpu().tolist()
                for i, corr_values in enumerate(zip(*correl)):
                    correlations_[i].extend(corr_values)

                # Compute errors
                errors, minimum, min_index, maximum, max_index = absolute_errors(x, x_recon)
                if isinstance(errors, torch.Tensor):
                    errors = errors.cpu().tolist()
                for i, error_values in enumerate(zip(*errors)):
                    rel_errors[i].extend(error_values)

                # Compute RMSE
                rmse.extend(get_RMSE(x,x_recon).tolist())
                
                #Track best reconstruction
                if minimum < minimum_error:
                    minimum_error = minimum
                    if 0 <= min_index < x.shape[0]:
                        x_best_input = x[min_index].cpu()
                        x_best_recon = x_recon[min_index].cpu()
                    else:
                        print(f"Warning: min_index {min_index} out of range for batch {batch_idx}")
                
                # Track worst reconstruction
                if maximum > max_error:
                    max_error = maximum
                    if 0<= max_index < x.shape[0]:
                        x_worst_input = x[max_index].cpu()
                        x_worst_recon = x_recon[max_index].cpu()
                    
            # Track first sample in each batch
            if len(x_input_N) <= N:
                x_input_N.append(x[0])
                x_recon_N.append(x_recon[0])
                        
    try:
        with open(os.path.join(output_directory , 'test_loss.json'), 'w') as f:
            data = {
                'loss_vs_batch':  loss_vs_batch,
                'best_loss': best_loss,
                'input' : x_best_input.flatten().numpy().tolist(),
                'reconstructed': x_best_recon.flatten().numpy().tolist()
            }
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"{e}")
        
    try:  
        fig, axs = plt.subplots(2, figsize=(8, 6))

        axs[0].plot(x_best_input.flatten().numpy().tolist(), label="Original", lw=2)
        axs[0].plot(x_best_recon.flatten().numpy().tolist(), label=f"Reconstructed ({minimum_error})", lw=2, linestyle="--")
        axs[0].set_title(f"{this_signal} Reconstruction (Min Error)")
        axs[0].set_xlabel("Time")  # X-axis label
        axs[0].legend()

        axs[1].plot(x_worst_input.flatten().numpy().tolist(), label="Original", lw=2)
        axs[1].plot(x_worst_recon.flatten().numpy().tolist(), label=f"Reconstructed ({max_error})", lw=2, linestyle="--")
        axs[1].set_title(f"{this_signal} Reconstruction (Max Error)")
        axs[1].set_xlabel("Time")  # X-axis label
        axs[1].legend()

        plt.savefig(output_dir + f"{this_signal}_flattened_reconstruction.pdf", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"{e}")
    
    try:
       
        # Prepare data
        x_best_in = x_best_input.squeeze(0)   
        x_best_re = x_best_recon.squeeze(0)
        x_worst_in = x_worst_input.squeeze(0)
        x_worst_re = x_worst_recon.squeeze(0)

        # Compute global min and max for consistent colour scale
        vmin = min(x_best_in.min(), x_best_re.min(), x_worst_in.min(), x_worst_re.min())
        vmax = max(x_best_in.max(), x_best_re.max(), x_worst_in.max(), x_worst_re.max())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
        axes = axes.flatten()

        # Best reconstruction
        axes[0].imshow(x_best_in, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{this_signal} Original (Best)")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Channel")

        axes[1].imshow(x_best_re, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"{this_signal} Reconstructed ({minimum_error:.4f})")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Channel")

        # Worst reconstruction
        axes[2].imshow(x_worst_in, vmin=vmin, vmax=vmax)
        axes[2].set_title(f"{this_signal} Original (Worst)")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Channel")

        im3 = axes[3].imshow(x_worst_re, vmin=vmin, vmax=vmax)
        axes[3].set_title(f"{this_signal} Reconstructed ({max_error:.4f})")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("Channel")

        # colourbar
        fig.colorbar(im3, ax=axes.ravel().tolist(), location='right', shrink=0.8, label='Amplitude')
        
        plt.savefig(output_dir + f"{this_signal}_image_reconstruction.pdf")

    except Exception as e:
        print(f"{e}")

    plot_histograms(
        correlations_,
        100,
        'blue',
        x_label="Correlations",
        y_label="frequency",
        title_prefix=f'',
        file_name=f'{output_dir}{this_signal}_correlations.pdf')
    
    plot_histograms(
        rel_errors,
        100,
        'red',
        x_label="Relative absolute errors",
        y_label="frequency",
        title_prefix=f'',
        file_name= f'{output_dir}{this_signal}_rel_errors.pdf',
        x_max = 1,
        x_min = 0)
    
    signal = this_signal
    file_path = output_dir
    
    with open(os.path.join(file_path, "loss_curves.json"), 'r') as file:
        data = json.load(file)

    # Validation loss values
    beta = SETTINGS.BETA_VAE.beta
    val_loss = data["Loss"]["val_total"]
    val_recon_loss = data["Loss"]["val_recon"]
    val_kl_loss  =  np.array(data["Loss"]["val_kl"])*beta
    train_loss = data["Loss"]["train_total"]
    train_recon_loss = data["Loss"]["train_recon"]
    train_kl_loss =  np.array(data["Loss"]["train_kl"])*beta
    # Epochs
    epochs = list(range(1, len(val_loss) + 1))

    patience = 5
    y = np.array(val_loss[-patience:])
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0] 
    print(f"Fit slope of {patience} last validation losses : {slope}")
        
    # Create scatter plot
    fig, ax = plt.subplots()
    ax.plot(epochs, val_loss, linestyle='solid',color='blue', marker='o', label="Validation total" )
    ax.plot(epochs, val_recon_loss, linestyle='dashed', color='blue', label="Validation recon")
    ax.plot(epochs, val_kl_loss, linestyle='dotted', color='blue', label=f"Validation kl * {beta}")
    ax.plot(epochs, train_loss, linestyle='solid',color='red', marker='o', label="Training total")
    ax.plot(epochs, train_recon_loss, linestyle='dashed',color='red', label="Training recon")
    ax.plot(epochs, train_kl_loss, linestyle='dotted',color='red', label=f"Training kl * {beta}")
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(signal + 'Conv1d_VAE')
    ax.grid(True)
    ax.legend()

    fig.savefig(file_path + '/losses_vs_batch.pdf')

    # Total loss
    loss = loss_vs_batch
    min_loss = min(loss)
    max_loss = max(loss)

    try:
        bins = np.arange(min_loss, 0.1 + 1e-4, 1e-4)
    except Exception as e:
        print(f"Error creating bins: {e}")
        bins = 100  # fallback to default number of bins

    fig, ax = plt.subplots()
    ax.hist(loss, bins=bins)
    ax.set_xlabel('Validation total loss')
    ax.set_yscale('log')
    ax.set_title(signal + "total loss")
    ax.legend([f'Batches: {len(loss)}'])
    plt.show()
    fig.savefig(file_path + f"/{this_signal}_TotalLoss.pdf")

    # RMSE
    fig, ax = plt.subplots()
    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.arange(min_loss, max_rmse + 1e-2, 1e-2)

    fig, ax = plt.subplots()
    ax.hist(rmse, bins=bins)
    ax.set_xlabel('RMSE')
    ax.set_yscale('log')
    ax.legend([f'Items: {len(rmse)}'])
    ax.set_title(signal + "RMSE")
    fig.savefig(file_path + f"/{this_signal}_RMSE.pdf")
 

    # Plot N reconstructions
    x_input_N = torch.stack(x_input_N)
    x_recon_N = torch.stack(x_recon_N)
            
    fig, axs = plt.subplots(N, figsize=(12,N*4))

    for i in range(N):
        try:
            # Flatten features and length into 1D
            original_flat = x_input_N[i].cpu().flatten().numpy().tolist()
            recon_flat = x_recon_N[i].cpu().flatten().numpy().tolist()

            # Plot original vs reconstructed
            axs[i].plot(original_flat, label="Original", lw=2)
            axs[i].plot(recon_flat, label="Reconstructed", lw=2, linestyle="--")
            axs[i].set_title(f"Sample {i+1} Original vs Recon")
            axs[i].set_xlabel("Time")
            axs[i].legend()
        except Exception as e:
            print(f"e")

    plt.savefig(output_dir + "/batch_reconstruction_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()

   
def get_RMSE(data, reco, eps = 1e-8):
    return torch.sqrt(torch.mean((data - reco) ** 2, dim=(1, 2)))

def correlations(data, reco, eps = 1e-8):
    """Compute time correlations for each feature 
    in data-reco pairs

    Parameters
    ----------
    data : tensor
        [batch, features, time]
    reco : _type_
        [batch, features, time]

    Returns
    -------
    Tensor or time correlations
        [batch, features]
    """
    # Subtract mean along time axis
    input_diff = data - data.mean(dim=-1, keepdim=True)
    reco_diff = reco - reco.mean(dim=-1, keepdim=True)

    # Compute numerator and denominator along time axis
    numerator = torch.sum(input_diff * reco_diff, dim=-1)  # [batch, features]
   
    denominator = torch.sqrt(torch.sum(input_diff ** 2, dim=-1) * torch.sum(reco_diff ** 2, dim=-1))  + eps # [batch, features]

    corr = numerator / denominator  # [batch, features]
    
    return corr

def absolute_errors(data, reco, eps = 1e-8):
    """Compute time-averaged absolute error for each feature 
    in data-reco pairs

    Parameters
    ----------
    data : tensor
        [batch, features, time]
    reco : _type_
        [batch, features, time]

    Returns
    -------
    Tensor
    time averaged absolute errors for each features in data-reco pairs
        [batch, features]
    Tensor [batch]
        minimum in the time- and features- averaged absolute error
    int
        index of the minimum in [batch]
    """

    abs_rel_error = torch.abs(data - reco)  # [batch, features, time]

    # Mean over time dimension
    time_averaged_rel_errors = abs_rel_error.mean(dim=-1)  # [batch, features]
    rel_error_per_sample = time_averaged_rel_errors.mean(dim =-1) # [batch]
    
    min_vals, min_index = torch.min(rel_error_per_sample, dim = 0) 
    max_vals, max_index = torch.max(rel_error_per_sample, dim = 0) 
    return time_averaged_rel_errors, min_vals.item(), min_index.item(), max_vals, max_index


if __name__ == "__main__":
    
    conf_file_name = "config_flux_loop_flux"
    directory_name = "conv1d_vae_"+conf_file_name
    output_dir = "scripts/pipelines/data/output/" + f"{directory_name}/"
    
    SETTINGS = get_settings(output_dir + f"{conf_file_name}.json")
    
    source, signal_name = SETTINGS.DATA.data_names[0]

    test_model(source, signal_name, output_dir, SETTINGS)