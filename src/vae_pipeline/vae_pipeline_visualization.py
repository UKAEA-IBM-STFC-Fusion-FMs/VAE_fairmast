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
if REPO_ROOT not in sys.path:sys.path.insert(0, REPO_ROOT)
from fairmast_tools.MAST_tools.MAST_dataset import MastDataset, CachedDataset
from src.vae_pipeline.utils.utils import ComposeTransforms
from src.vae_pipeline.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import StdScalingTransform
from src.vae_pipeline.transforms.shot_level_transforms.window_segmenter_transform import WindowSegmenterTransform
from src.vae_pipeline.transforms.signal_level_transforms.imputer_transform import ImputerTransform
from src.vae_pipeline.configs.config_setup import get_settings
from src.vae_pipeline.models.vae_model import loss_function_batch_mean as loss_function
from src.vae_pipeline.models.vae_model import beta_VAE
from src.vae_pipeline.transforms.shot_level_transforms.conv1d_vae_transform import Conv1dVAETransform
from src.vae_pipeline.collate_functions.collate_functions import  WindowsCollate
from src.vae_pipeline.utils.utils import get_train_test_val_shots
from src.vae_pipeline.vae_pipeline import initialize_datasets, initialize_dataloaders
from src.vae_pipeline.transforms.shot_level_transforms.concatenate_signals_transform import ConcatenateSignalsAfterTimeSegmentation

# Determine device to train on
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--------------- RUNNING ON GPUs ---------------")
else:
    device = torch.device("cpu")
    print(f"--------------- RUNNING ON CPUs ---------------")
    

def plot_histograms(
    properties,
    color,
    x_label,
    y_label,
    title_prefix,
    file_name,
    num_rows=3,
    num_cols=4
):
    num_features = len(properties)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))

    # Normalize axes to a flat list
    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = axes.ravel().tolist()

    max_plots = len(axes_list)
    plots_to_draw = min(num_features, max_plots)

    # Plot
    for i in range(plots_to_draw):
        ax = axes_list[i]
        data = properties[i]

        q95 = float(np.quantile(data,0.95))
        
        min_data = min(data)
        max_data = float(np.quantile(data,0.9973))
        
        bins = np.linspace(min_data, max_data, 100)
        ax.hist(data, bins=bins, color=color, alpha=0.7, range=(min_data, max_data))
        ax.axvline(q95, color='green', linestyle='--', linewidth=1.5, label=f'95% threshold: {q95:.4g}')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend([f"Ch. {i+1}: {len(data)} items", f'95% threshold: {q95:.4g}'])

    # Hide unused subplots
    for j in range(plots_to_draw, max_plots):
        axes_list[j].axis('off')

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

 
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
    model_path = os.path.join(output_dir, "best_vae_" + signal_name + ".pt")
     
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
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, _, val_shots = get_train_test_val_shots(
        max_index_for_train = SETTINGS.TRAINING.num_train_samples,
        max_index_for_val = SETTINGS.TRAINING.num_val_samples,
        max_index_for_test = None
    )

    this_signal = signal_name
    
    # Get mean and std for signal transformation
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_mean_shot.pkl"), "rb") as f:
        dict_mean = pickle.load(f)
    with open(os.path.join(SETTINGS.LOCAL_PATHS.global_mean_std_path, "dict_std_shot.pkl"), "rb") as f:
        dict_std = pickle.load(f)
        
    # Signal-level transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [   
                StdScalingTransform(dict_mean[var], dict_std[var]),
                ImputerTransform(),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform map
    if len(SETTINGS.DATA.data_names)>1: # Merge signals
        print("WARNING: current pipeline supports single signal analysis only.\
            All signals in the list will be merged into one, if compatible")
        shot_transforms = ComposeTransforms(
            [
                WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
                ConcatenateSignalsAfterTimeSegmentation(),
                Conv1dVAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window),
            ]
        )
    else:
        shot_transforms = ComposeTransforms(
            [
                WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
                Conv1dVAETransform(SETTINGS.TIME_SEGMENTATION.targeted_time_stamps_per_window),
            ]
        )
    
    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots},
        signal_transform_map=signal_transform_map,
        shot_transforms=shot_transforms,
        local_flag=SETTINGS.DATA.local,
        cache_data=False
    )

    conv1d_vae_collate_fn =  WindowsCollate()
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=conv1d_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    val_dataloader = dataloaders_train_val_test["val"]
    
    # Create conv1d-VAE models
    model = beta_VAE(SETTINGS)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Epoch of the best model: {checkpoint['epoch']}")
    
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

    
    # with torch.no_grad(): 
    #     for batch_idx, batch in enumerate(val_dataloader):
            
    #         print(f"Batch idx {batch_idx}")
            
    #         if batch is None:
    #             continue
            
            
    #         # if batch_idx==1:
    #         #     break
                
    #         for group_idx, stacked_tensor in batch.items():
                
    #             # if group_idx ==1:
    #             #     break
                
    #             x = stacked_tensor.to(device)
    #             x_recon, mu, logvar = model(x)

    #             # Compute loss
    #             total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x, mu, logvar)
    #             loss_vs_batch.append(total_loss.item())  
                
    #             if total_loss.item() < best_loss:
    #                 best_loss = total_loss.item()
                    
    #             num_channels = x.shape[1]
    #             if not correlations_:
    #                 correlations_ = [[] for _ in range(num_channels)]
    #             if not rel_errors:
    #                 rel_errors = [[] for _ in range(num_channels)]

    #             # Compute correlations
    #             correl = correlations(x, x_recon)
    #             if isinstance(correl, torch.Tensor):
    #                 correl = correl.cpu().tolist()
    #             for i, corr_values in enumerate(zip(*correl)):
    #                 correlations_[i].extend(corr_values)

    #             # Compute errors
    #             errors, minimum, min_index, maximum, max_index = time_averaged_absolute_errors(x, x_recon)
    #             if isinstance(errors, torch.Tensor):
    #                 errors = errors.cpu().tolist()
    #             for i, error_values in enumerate(zip(*errors)):
    #                 rel_errors[i].extend(error_values)

    #             # Compute RMSE
    #             rmse.extend(get_RMSE(x,x_recon).tolist())
                
    #             #Track best reconstruction
    #             if minimum < minimum_error:
    #                 minimum_error = minimum
    #                 if 0 <= min_index < x.shape[0]:
    #                     x_best_input = x[min_index].cpu()
    #                     x_best_recon = x_recon[min_index].cpu()
    #                 else:
    #                     print(f"Warning: min_index {min_index} out of range for batch {batch_idx}")
                
    #             # Track worst reconstruction
    #             if maximum > max_error:
    #                 max_error = maximum
    #                 if 0<= max_index < x.shape[0]:
    #                     x_worst_input = x[max_index].cpu()
    #                     x_worst_recon = x_recon[max_index].cpu()
                    
    #         # Track first sample in each batch
    #         if len(x_input_N) <= N:
    #             x_input_N.append(x[0])
    #             x_recon_N.append(x_recon[0])
    
    sub_batch_size = SETTINGS.TRAINING.train_batch_size            
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(val_dataloader):
            
            x = batch["x"]
            
            if x.numel() == 0:
                continue  # skip empty batch
        
            
            print(f"Batch idx: {batch_idx}")

            total_tensors = x.size(0)
            
            for start in range(0, total_tensors, sub_batch_size):
                end = min(start + sub_batch_size, total_tensors)
            
                x_sub_batch = x[start:end].to(device)
                sub_tensors = x_sub_batch.size(0)
                    
                x_recon, mu, logvar = model(x_sub_batch)
                                        
                # Compute loss
                try:
                    total_loss, recon_loss, kl_loss = loss_function(SETTINGS.BETA_VAE.beta, x_recon, x_sub_batch, mu, logvar,  clamp_logvar=(-20.0, 20.0),clamp_mu=(None,None))
            
                    total_loss = torch.nan_to_num(total_loss,nan=0.0,posinf=None,neginf=None) 
                    recon_loss = torch.nan_to_num(recon_loss,nan=0.0,posinf=None,neginf=None) 
                    kl_loss = torch.nan_to_num(kl_loss,nan=0.0,posinf=None,neginf=None) 
                    loss_vs_batch.append(total_loss.item()) 
                    
                except ValueError as e:
                    print(f"Error in loss function calculation: {e}")
                    continue
                
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    
                num_channels = x_sub_batch.shape[1]
                if not correlations_:
                    correlations_ = [[] for _ in range(num_channels)]
                if not rel_errors:
                    rel_errors = [[] for _ in range(num_channels)]

                # Compute correlations
                correl = correlations(x_sub_batch, x_recon)
                if isinstance(correl, torch.Tensor):
                    correl = correl.cpu().tolist()
                for i, corr_values in enumerate(zip(*correl)):
                    correlations_[i].extend(corr_values)

                # Compute errors
                errors, minimum, min_index, maximum, max_index = time_averaged_absolute_errors(x_sub_batch, x_recon)
                if isinstance(errors, torch.Tensor):
                    errors = errors.cpu().tolist()
                for i, error_values in enumerate(zip(*errors)):
                    rel_errors[i].extend(error_values)

                # Compute RMSE
                rmse.extend(get_RMSE(x_sub_batch,x_recon).tolist())
                
                #Track best reconstruction
                if minimum < minimum_error:
                    minimum_error = minimum
                    if 0 <= min_index < x_sub_batch.shape[0]:
                        x_best_input = x_sub_batch[min_index].cpu()
                        x_best_recon = x_recon[min_index].cpu()
                    else:
                        print(f"Warning: min_index {min_index} out of range for batch {batch_idx}")
                
                # Track worst reconstruction
                if maximum > max_error:
                    max_error = maximum
                    if 0<= max_index < x_sub_batch.shape[0]:
                        x_worst_input = x_sub_batch[max_index].cpu()
                        x_worst_recon = x_recon[max_index].cpu()
                    
            # Track first sample in each batch
            if len(x_input_N) <= N:
                x_input_N.append(x_sub_batch[0])
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
        axs[0].plot(x_best_recon.flatten().numpy().tolist(), label=f"Reconstructed ({minimum_error:.4f})", lw=2, linestyle="--")
        axs[0].set_title(f"{this_signal} Reconstruction (Min Error)")
        axs[0].set_xlabel("Time")  # X-axis label
        axs[0].legend()

        axs[1].plot(x_worst_input.flatten().numpy().tolist(), label="Original", lw=2)
        axs[1].plot(x_worst_recon.flatten().numpy().tolist(), label=f"Reconstructed ({max_error:.2f})", lw=2, linestyle="--")
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
        'blue',
        x_label="Correlations",
        y_label="frequency",
        title_prefix=f'',
        file_name=f'{output_dir}{this_signal}_correlations.pdf')
    
    plot_histograms(
        rel_errors,
        'red',
        x_label="Relative absolute errors",
        y_label="frequency",
        title_prefix=f'',
        file_name= f'{output_dir}{this_signal}_rel_errors.pdf')
    
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
    max_loss = float(np.quantile(loss, 0.9973))

    try:
        bins = np.linspace(min_loss, max_loss, 200)
    except Exception as e:
        print(f"Error creating bins: {e}")
        bins = 100  # fallback to default number of bins

    p95_loss = float(np.quantile(loss, 0.95))  # 95th percentile
    
    fig, ax = plt.subplots()
    ax.hist(loss, bins=bins)
    ax.axvline(p95_loss, color='red', linestyle='--', linewidth=1.5, label=f'95% threshold: {p95_loss:.4g}')
    ax.set_xlabel('Validation total loss')
    ax.set_yscale('log')
    ax.set_title(signal + "total loss")
    ax.legend([f'Batches: {len(loss)}', f'95% threshold: {p95_loss:.4g}'])
    plt.show()
    fig.savefig(file_path + f"/{this_signal}_TotalLoss.pdf")

    # RMSE
    fig, ax = plt.subplots()
    min_rmse = min(rmse)
    max_rmse = max(rmse)
    bins = np.linspace(min_loss, max_rmse, 200)
    p95_rmse = float(np.quantile(rmse, 0.95))
    
    fig, ax = plt.subplots()
    ax.hist(rmse, bins=bins)
    ax.axvline(p95_rmse, color='red', linestyle='--', linewidth=1.5, label=f'95% threshold: {p95_rmse:.4g}')
    ax.set_xlabel('RMSE')
    ax.set_yscale('log')
    ax.legend([f'Items: {len(rmse)}', f'95% threshold: {p95_rmse:.4g}'])
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

   
def get_RMSE(data, reco):
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

def time_averaged_absolute_errors(data, reco):
    """Compute time-averaged absolute error for each feature 
    in data-reco pairs

    Parameters
    ----------
    data : tensor
        [batch, features, time]
    reco : tensor
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

    abs_error = torch.abs(data - reco)  # [batch, features, time]

    # Mean over time dimension
    time_averaged_errors = abs_error.mean(dim=-1)  # [batch, features]
    rel_error_per_sample = time_averaged_errors.mean(dim =-1) # [batch]
    
    min_vals, min_index = torch.min(rel_error_per_sample, dim = 0) 
    max_vals, max_index = torch.max(rel_error_per_sample, dim = 0) 
    return time_averaged_errors, min_vals.item(), min_index.item(), max_vals, max_index


if __name__ == "__main__":
    
    conf_file_name = "config_coil_current_3"
    directory_name = "conv1d_vae_"+conf_file_name
    output_dir = "src/vae_pipeline/data/output/" + f"{directory_name}/"
    
    SETTINGS = get_settings(output_dir + f"{conf_file_name}.json")
    
    source, signal_name = SETTINGS.DATA.data_names[0]

    test_model(source, signal_name, output_dir, SETTINGS)