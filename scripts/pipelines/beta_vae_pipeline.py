import os
import sys
import pickle
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict
import json
import argparse
import shutil

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(f"REPO_ROOT: {REPO_ROOT}")

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.utils.utils import (
    read_data_split_csv, ComposeTransforms
)
from scripts.pipelines.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipelines.preprocessing.standardscaling_preprocessing import (
    get_mean_shot,
    get_std_shot,
)
from scripts.pipelines.transforms.signal_level_transforms.fill_with_zeros_imputer_transform import (
    FillWithZerosImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.forward_fill_imputer_transform import (
    ForwardFillImputerTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform,
)
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
    SamplingToReferenceTimeTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import (
    TruncationTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import (
    WindowSegmenterTransform,
)
from scripts.pipelines.transforms.shot_level_transforms.beta_vae_transform import (
    BetaVAETransform,
)
from scripts.pipelines.models.beta_vae_model import BetaVAE
from scripts.pipelines.configs.config_setup import get_settings
from scripts.pipelines.collate_functions.collate_functions import beta_vae_collate_fn

print(f"\nNumber of Cores: {cpu_count()}\n")

# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_train_test_val_shots(max_index=None,num_train=None, num_val=None,num_test=None):
    train_sh, test_sh, val_sh = read_data_split_csv()

    if max_index:
        np.random.shuffle(train_sh)
        np.random.shuffle(val_sh)
        np.random.shuffle(test_sh)
        train_sh = train_sh[0:max_index]
        val_sh = val_sh[0:max_index]
        test_sh = test_sh[0:max_index]
    if num_train:
        np.random.shuffle(train_sh)
        train_sh = train_sh[0:num_train]
    if num_val:
        np.random.shuffle(val_sh)
        val_sh = val_sh[0:num_val]
    if num_test:
        np.random.shuffle(test_sh)
        test_sh = test_sh[0:num_test]

    return train_sh, test_sh, val_sh


def fit_mean_and_std_for_signal_transform(
    output_sub_dir, verbose=False, use_existing=False
):
    """
    Fit or load mean and std for signal transformation.

    Args:
        output_sub_dir: Directory to save/load fitted parameters
        verbose: Print verbose output
        use_existing: If True, try to load existing fitted parameters instead of re-fitting
    """
    output_dir = os.path.join("output", output_sub_dir)
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
        local=LOCAL_FLAG,
        shots_list=train_shots,
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


def initialize_datasets(
    sources_and_signals, shots, sig_tran_map, shot_tran, local_flag=False, verbose=False
):
    datasets_ = {"train": None, "val": None, "test": None}

    # Train
    if shots["train"]:
        datasets_["train"] = MastDataset(
            local=local_flag,
            shots_list=shots["train"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(mast_train_dataset): {len(datasets_['train'])}")

    # Val
    if shots["val"]:
        datasets_["val"] = MastDataset(
            local=local_flag,
            shots_list=shots["val"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(val_dataset): {len(datasets_['val'])}")

    # Test
    if shots["test"]:
        datasets_["test"] = MastDataset(
            local=local_flag,
            shots_list=shots["test"],
            source_signal_list=sources_and_signals,
            signal_level_transform_map=sig_tran_map,
            shot_level_transform=shot_tran,
        )
        if verbose:
            print(f"len(test_dataset): {len(datasets_['test'])}")

    return datasets_


def initialize_dataloaders(
    datasets,
    collate_function,
    batch_size,
    num_workers,
    shuffle=True,
    drop_last=False,
    verbose=False,
):
    dataloaders_ = {"train": None, "val": None, "test": None}

    if verbose:
        print("\n\n----------DATASET & DATALOADER INITIALIZATION----------\n")

    # Train
    if datasets["train"]:
        dataloaders_["train"] = DataLoader(
            dataset=datasets["train"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    # Val
    if datasets["val"]:
        dataloaders_["val"] = DataLoader(
            dataset=datasets["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    # Test
    if datasets["test"]:
        dataloaders_["test"] = DataLoader(
            dataset=datasets["test"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_function,
        )

    return dataloaders_


def create_beta_vae_models(
    train_dataloader_, 
    latent_dim,
    hidden_dim,
    beta,
    verbose=False
    ):
    """Create β-VAE models for each signal type"""
    if verbose:
        print("\n\n----------β-VAE MODEL INITIALIZATION----------\n")

    # Get sample batch to determine signal shapes
    sample_batch = next(iter(train_dataloader_))

    models = {}
    for signal_name, signal_data in sample_batch.items():
        input_length = signal_data.shape[-1]  # Last dimension is time

        if verbose:
            print(
                f"Signal: {signal_name}, Shape: {signal_data.shape}, Input length: {input_length}"
            )

        model = BetaVAE(input_length=input_length, latent_dim=latent_dim, hidden_dim=hidden_dim,beta=beta).to(
            device
        )

        models[signal_name] = model

        if verbose:
            print(f"Created BetaVAE for {signal_name}")

    return models


def train_beta_vae_models(
    models, train_dataloader, val_dataloader, output_sub_dir, verbose=False,use_cosineLr=False,cos_lr_min = 10**-6
):
    """Train β-VAE models for each signal"""
    if verbose:
        print("\n\n----------β-VAE TRAINING----------\n")

    output_dir = os.path.join("output", output_sub_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create optimizers for each model
    optimizers = {}
    for signal_name, model in models.items():
        optimizers[signal_name] = torch.optim.Adam(model.parameters(), lr=SETTINGS.BETA_VAE.lr)
        if use_cosineLr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[signal_name], T_max=SETTINGS.TRAINING.num_epochs,eta_min = cos_lr_min)

    # Training tracking
    best_losses = {signal_name: float("inf") for signal_name in models.keys()}
    best_model_states = {}
    
    # Loss tracking
    loss_curves = {}
    for signal_name in models.keys():
        loss_curves[signal_name] = {
            'train_total': [],
            'train_recon': [],
            'train_kl': [],
            'val_total': [],
            'val_recon': [],
            'val_kl': []
        }

    for epoch in range(SETTINGS.TRAINING.num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}\n")

        # Training phase
        for signal_name, model in models.items():
            model.train()

        train_losses = defaultdict(float)
        train_recon_losses = defaultdict(float)
        train_kl_losses = defaultdict(float)
        train_counts = defaultdict(int)

        for signal_name in models.keys():
            train_losses[signal_name] = 0.
            train_recon_losses[signal_name] = 0.
            train_kl_losses[signal_name] = 0.
            train_counts[signal_name] = 0

        if verbose:
            print("Training phase")

        for batch_idx, batch_signals in enumerate(train_dataloader):
            if verbose:
                print(f"Batch idx: {batch_idx}")
            if verbose and batch_idx == 0:
                print(f"Available signals in batch: {list(batch_signals.keys())}")

            for signal_name, signal_data in batch_signals.items():
                if signal_name not in models:
                    continue

                model = models[signal_name]
                optimizer = optimizers[signal_name]

                # Prepare data
                if signal_data.dim() == 2:  # (batch, time)
                    x = signal_data.to(device)
                else:  # (batch, channels, time) -> flatten channels
                    x = signal_data.view(-1, signal_data.size(-1)).to(device)

                # Forward pass
                reconstruction, mu, logvar, z = model(x)
                total_loss, recon_loss, kl_loss = model.loss_function(
                    reconstruction, x, mu, logvar
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_losses[signal_name] += total_loss.item()
                train_recon_losses[signal_name] += recon_loss.item()
                train_kl_losses[signal_name] += kl_loss.item()
                train_counts[signal_name] += x.size(0)

        # Validation phase
        val_losses = defaultdict(float)
        val_recon_losses = defaultdict(float)
        val_kl_losses = defaultdict(float)
        val_counts = defaultdict(int)
        for signal_name in models.keys():
            val_losses[signal_name] = 0.
            val_recon_losses[signal_name] = 0.
            val_kl_losses[signal_name] = 0.
            val_counts[signal_name] = 0

        for signal_name, model in models.items():
            model.eval()

        if verbose:
            print("\nValidation phase")

        with torch.no_grad():
            for batch_idx, batch_signals in enumerate(val_dataloader):
                if verbose:
                    print(f"Batch idx: {batch_idx}")
                for signal_name, signal_data in batch_signals.items():
                    if signal_name not in models:
                        continue

                    model = models[signal_name]

                    # Prepare data
                    if signal_data.dim() == 2:
                        x = signal_data.to(device)
                    else:
                        x = signal_data.view(-1, signal_data.size(-1)).to(device)

                    # Forward pass
                    reconstruction, mu, logvar, z = model(x)
                    total_loss, recon_loss, kl_loss = model.loss_function(
                        reconstruction, x, mu, logvar
                    )

                    val_losses[signal_name] += total_loss.item()
                    val_recon_losses[signal_name] += recon_loss.item()
                    val_kl_losses[signal_name] += kl_loss.item()
                    val_counts[signal_name] += x.size(0)

        # Store loss curves and print epoch results
        for signal_name in models.keys():
            if train_counts[signal_name] > 0:
                avg_train_loss = train_losses[signal_name] / train_counts[signal_name]
                avg_train_recon = train_recon_losses[signal_name] / train_counts[signal_name]
                avg_train_kl = train_kl_losses[signal_name] / train_counts[signal_name]
                
                if val_counts[signal_name] > 0:
                    avg_val_loss = val_losses[signal_name] / val_counts[signal_name]
                    avg_val_recon = val_recon_losses[signal_name] / val_counts[signal_name]
                    avg_val_kl = val_kl_losses[signal_name] / val_counts[signal_name]
                else:
                    avg_val_loss = float("inf")
                    avg_val_recon = float("inf")
                    avg_val_kl = float("inf")

                # Store loss curves
                loss_curves[signal_name]['train_total'].append(avg_train_loss)
                loss_curves[signal_name]['train_recon'].append(avg_train_recon)
                loss_curves[signal_name]['train_kl'].append(avg_train_kl)
                loss_curves[signal_name]['val_total'].append(avg_val_loss)
                loss_curves[signal_name]['val_recon'].append(avg_val_recon)
                loss_curves[signal_name]['val_kl'].append(avg_val_kl)

                if verbose:
                    print(
                        f"Signal {signal_name:30s} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )

                # Save best model
                if avg_val_loss < best_losses[signal_name]:
                    best_losses[signal_name] = avg_val_loss
                    best_model_states[signal_name] = models[signal_name].state_dict()

                    # Save best model state
                    model_path = os.path.join(
                        output_dir, f"best_beta_vae_{signal_name.replace('/', '_')}.pt"
                    )
                    torch.save(best_model_states[signal_name], model_path)

    return best_model_states, loss_curves


def visualize_beta_vae_results(
    models, train_dataloader, val_dataloader, output_sub_dir, loss_curves, verbose=False
):
    """Create visualizations for all trained β-VAE models"""
    if verbose:
        print("\n\n----------β-VAE VISUALIZATION----------\n")
    
    output_dir = os.path.join("output", output_sub_dir)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for model in models.values():
        model.eval()
    
    with torch.no_grad():
        # Get sample batches for visualization
        train_batch = next(iter(train_dataloader))
        val_batch = next(iter(val_dataloader)) if val_dataloader else train_batch
        
        for signal_name, model in models.items():
            if verbose:
                print(f"Creating visualizations for signal: {signal_name}")
            
            # Get signal data from both train and validation
            train_signal_data = train_batch.get(signal_name)
            val_signal_data = val_batch.get(signal_name)
            
            if train_signal_data is None:
                if verbose:
                    print(f"No data found for signal {signal_name}, skipping")
                continue
            
            # Prepare data (flatten if needed)
            if train_signal_data.dim() == 2:
                train_x = train_signal_data.to(device)
            else:
                train_x = train_signal_data.view(-1, train_signal_data.size(-1)).to(device)
            
            if val_signal_data is not None:
                if val_signal_data.dim() == 2:
                    val_x = val_signal_data.to(device)
                else:
                    val_x = val_signal_data.view(-1, val_signal_data.size(-1)).to(device)
            else:
                val_x = train_x
            
            # Get reconstructions and latent representations
            train_recon, train_mu, train_logvar, train_z = model(train_x)
            val_recon, val_mu, val_logvar, val_z = model(val_x)
            print("train_x.size()",train_x.size())
            print("np.shape(train_x[0].cpu().numpy())",np.shape(train_x[0].cpu().numpy()))
            print("train_recon.size()",train_recon.size())
            print("np.shape(train_recon[0].cpu().numpy())",np.shape(train_recon[0].cpu().numpy()))
            
            # Calculate losses for display
            train_loss, train_recon_loss, train_kl_loss = model.loss_function(
                train_recon, train_x, train_mu, train_logvar
            )
            val_loss, val_recon_loss, val_kl_loss = model.loss_function(
                val_recon, val_x, val_mu, val_logvar
            )
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 4, figsize=(20, 12))
            
            # 1. Original vs Reconstructed signals (training data)
            ax1 = axes[0, 0]
            n_samples_to_show = min(5, train_x.shape[0])
            for i in range(n_samples_to_show):
                offset = i * 2
                ax1.plot(train_x[i].cpu().numpy() + offset, 'b-', alpha=0.7,
                        label='Original' if i == 0 else '')
                ax1.plot(train_recon[i].cpu().numpy() + offset, 'r--', alpha=0.7,
                        label='Reconstructed' if i == 0 else '')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Amplitude (offset)')
            ax1.set_title(f'Train: Original vs Reconstructed\n{signal_name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Original vs Reconstructed signals (validation data)
            ax2 = axes[0, 1]
            n_samples_to_show = min(5, val_x.shape[0])
            for i in range(n_samples_to_show):
                offset = i * 2
                ax2.plot(val_x[i].cpu().numpy() + offset, 'b-', alpha=0.7,
                        label='Original' if i == 0 else '')
                ax2.plot(val_recon[i].cpu().numpy() + offset, 'r--', alpha=0.7,
                        label='Reconstructed' if i == 0 else '')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Amplitude (offset)')
            ax2.set_title(f'Val: Original vs Reconstructed\n{signal_name}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Reconstruction error distribution
            ax3 = axes[0, 2]
            train_error = torch.abs(train_recon - train_x).cpu().numpy().flatten()
            val_error = torch.abs(val_recon - val_x).cpu().numpy().flatten()
            ax3.hist(train_error, bins=50, alpha=0.7, label='Train Error', density=True)
            ax3.hist(val_error, bins=50, alpha=0.7, label='Val Error', density=True)
            ax3.set_xlabel('Absolute Error')
            ax3.set_ylabel('Density')
            ax3.set_title('Reconstruction Error Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Latent space visualization (first 2 dimensions)
            ax4 = axes[0, 3]
            ax4.scatter(train_mu[:, 0].cpu().numpy(), train_mu[:, 1].cpu().numpy(),
                       alpha=0.6, s=20, label='Train', c='blue')
            ax4.scatter(val_mu[:, 0].cpu().numpy(), val_mu[:, 1].cpu().numpy(),
                       alpha=0.6, s=20, label='Val', c='red')
            ax4.set_xlabel('Latent Dim 0')
            ax4.set_ylabel('Latent Dim 1')
            ax4.set_title('Latent Space (2D Projection)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Latent space variance (all dimensions)
            ax5 = axes[1, 0]
            latent_var_train = torch.var(train_z, dim=0).cpu().numpy()
            latent_var_val = torch.var(val_z, dim=0).cpu().numpy()
            dims = np.arange(len(latent_var_train))
            ax5.bar(dims - 0.2, latent_var_train, 0.4, label='Train', alpha=0.7)
            ax5.bar(dims + 0.2, latent_var_val, 0.4, label='Val', alpha=0.7)
            ax5.set_xlabel('Latent Dimension')
            ax5.set_ylabel('Variance')
            ax5.set_title('Latent Space Variance per Dimension')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Generated samples from random latent codes
            ax6 = axes[1, 1]
            random_z = torch.randn(5, model.latent_dim).to(device)
            generated_samples = model.decode(random_z)
            for i in range(5):
                ax6.plot(generated_samples[i].cpu().numpy() + i * 1.5,
                        label=f'Generated {i+1}')
            ax6.set_xlabel('Time Steps')
            ax6.set_ylabel('Amplitude (offset)')
            ax6.set_title('Samples from Random Latent Codes')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # 7. Loss components comparison
            ax7 = axes[1, 2]
            loss_data = {
                'Total Loss': [train_loss.item(), val_loss.item()],
                'Recon Loss': [train_recon_loss.item(), val_recon_loss.item()],
                'KL Loss': [train_kl_loss.item(), val_kl_loss.item()]
            }
            x_pos = np.arange(len(loss_data))
            width = 0.35
            
            train_losses = [loss_data[key][0] for key in loss_data.keys()]
            val_losses = [loss_data[key][1] for key in loss_data.keys()]
            
            ax7.bar(x_pos - width/2, train_losses, width, label='Train', alpha=0.7)
            ax7.bar(x_pos + width/2, val_losses, width, label='Val', alpha=0.7)
            ax7.set_xlabel('Loss Type')
            ax7.set_ylabel('Loss Value')
            ax7.set_title('Loss Components Comparison')
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(loss_data.keys(), rotation=45)
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            # 8. Training Loss Curves
            ax8 = axes[1, 3]
            if signal_name in loss_curves and len(loss_curves[signal_name]['train_total']) > 0:
                epochs = range(1, len(loss_curves[signal_name]['train_total']) + 1)
                
                # Plot total loss
                ax8.plot(epochs, loss_curves[signal_name]['train_total'], 'b-', 
                        label='Train Total', linewidth=2)
                ax8.plot(epochs, loss_curves[signal_name]['val_total'], 'b--', 
                        label='Val Total', linewidth=2)
                
                # Plot reconstruction loss
                ax8.plot(epochs, loss_curves[signal_name]['train_recon'], 'g-', 
                        label='Train Recon', alpha=0.7)
                ax8.plot(epochs, loss_curves[signal_name]['val_recon'], 'g--', 
                        label='Val Recon', alpha=0.7)
                
                # Plot KL loss
                ax8.plot(epochs, loss_curves[signal_name]['train_kl'], 'r-', 
                        label='Train KL', alpha=0.7)
                ax8.plot(epochs, loss_curves[signal_name]['val_kl'], 'r--', 
                        label='Val KL', alpha=0.7)
                
                ax8.set_xlabel('Epoch')
                ax8.set_ylabel('Loss Value')
                ax8.set_title('Training Loss Curves')
                ax8.legend(fontsize='small')
                ax8.grid(True, alpha=0.3)
                ax8.set_yscale('log')
            else:
                ax8.text(0.5, 0.5, 'No loss curves available', 
                        transform=ax8.transAxes, ha='center', va='center')
                ax8.set_title('Training Loss Curves')
            
            fig.tight_layout()
            
            # Save figure
            safe_signal_name = signal_name.replace('/', '_').replace('-', '_')
            fig_path = os.path.join(viz_dir, f'beta_vae_results_{safe_signal_name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            if verbose:
                print(f"Saved figure: {fig_path}")
        
        # Create a summary figure for all signals
        if len(models) > 1:
            create_multi_signal_summary(models, train_batch, val_batch, viz_dir, verbose)


def create_multi_signal_summary(models, train_batch, val_batch, viz_dir, verbose=False):
    """Create a summary visualization comparing all signals"""
    if verbose:
        print("Creating multi-signal summary visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    signal_names = []
    compression_ratios = []
    latent_dims = []
    train_losses = []
    val_losses = []
    
    with torch.no_grad():
        for signal_name, model in models.items():
            if signal_name not in train_batch:
                continue
                
            signal_names.append(signal_name)
            compression_ratios.append(model.input_length / model.latent_dim)
            latent_dims.append(model.latent_dim)
            
            # Get sample data and compute losses
            train_data = train_batch[signal_name]
            val_data = val_batch.get(signal_name, train_data)
            
            if train_data.dim() > 2:
                train_data = train_data.view(-1, train_data.size(-1))
            if val_data.dim() > 2:
                val_data = val_data.view(-1, val_data.size(-1))
                
            train_data = train_data.to(device)
            val_data = val_data.to(device)
            
            train_recon, train_mu, train_logvar, _ = model(train_data)
            val_recon, val_mu, val_logvar, _ = model(val_data)
            
            train_loss, _, _ = model.loss_function(train_recon, train_data, train_mu, train_logvar)
            val_loss, _, _ = model.loss_function(val_recon, val_data, val_mu, val_logvar)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
    
    # 1. Compression ratios comparison
    axes[0, 0].bar(range(len(signal_names)), compression_ratios, alpha=0.7)
    axes[0, 0].set_xlabel('Signal')
    axes[0, 0].set_ylabel('Compression Ratio')
    axes[0, 0].set_title('Compression Ratios by Signal')
    axes[0, 0].set_xticks(range(len(signal_names)))
    axes[0, 0].set_xticklabels([name.split('-')[-1][:10] for name in signal_names], 
                               rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Latent dimensions comparison
    axes[0, 1].bar(range(len(signal_names)), latent_dims, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Signal')
    axes[0, 1].set_ylabel('Latent Dimension')
    axes[0, 1].set_title('Latent Dimensions by Signal')
    axes[0, 1].set_xticks(range(len(signal_names)))
    axes[0, 1].set_xticklabels([name.split('-')[-1][:10] for name in signal_names], 
                               rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss comparison
    x_pos = np.arange(len(signal_names))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, train_losses, width, label='Train Loss', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, val_losses, width, label='Val Loss', alpha=0.7)
    axes[1, 0].set_xlabel('Signal')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].set_title('Final Losses by Signal')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name.split('-')[-1][:10] for name in signal_names], 
                               rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
β-VAE Training Summary:

Number of Signals: {len(signal_names)}
Average Compression Ratio: {np.mean(compression_ratios):.1f}:1
Average Latent Dimension: {np.mean(latent_dims):.1f}

Signal List:
"""
    for i, name in enumerate(signal_names):
        summary_text += f"• {name.split('-')[-1][:20]}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # Save summary figure
    summary_path = os.path.join(viz_dir, 'beta_vae_multi_signal_summary.png')
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved multi-signal summary: {summary_path}")


if __name__ == "__main__":

    # Initialize SETTINGS object
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_file",type=str,default="scripts/pipelines/configs/config_beta_vae.json")
    args=parser.parse_args()
    config_filename=args.config_file
    SETTINGS = get_settings(config_filename)
    
    LOCAL_FLAG = SETTINGS.DATA.local
    mp.set_start_method("spawn", force=True)

    # For common pipeline
    OUTPUT_SUB_FOLDER = SETTINGS.LOCAL_PATHS.data_output_directory + "beta_vae_output/"

    # Save input file to output folder
    output_dir = os.path.join("output", OUTPUT_SUB_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(config_filename,output_dir)



    source_signal_list = SETTINGS.DATA.data_names + SETTINGS.DATA.target_names

    # Parameters for window segmentation (no x/y split for VAE)
    PARAMETERS_WINDOWS_SEGMENTER = {
        "x_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.data_names],
        "y_keys": [f"{source}-{signal}" for source, signal in SETTINGS.DATA.target_names],  # Same as x for VAE
        "x_window_sec": SETTINGS.TIME_SEGMENTATION.x_window_sec,  # 100ms windows
        "y_window_sec": SETTINGS.TIME_SEGMENTATION.y_window_sec,
        "dt_sec": SETTINGS.TIME_SEGMENTATION.dt_sec, 
        "stride_sec": SETTINGS.TIME_SEGMENTATION.stride_sec,
        "stride_unitary": SETTINGS.TIME_SEGMENTATION.stride_unitary,
        "min_samples_per_window": SETTINGS.TIME_SEGMENTATION.min_samples_per_window,
        "verbose": False,
    }

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = get_train_test_val_shots(
        num_train=SETTINGS.TRAINING.num_train_samples,
        num_val=SETTINGS.TRAINING.num_val_samples,
        num_test=SETTINGS.TRAINING.num_test_samples
    )

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True,
        use_existing=SETTINGS.BETA_VAE.existing_fitted_params,
    )

    # Get the signal transform map
    signal_transform_map = {
        var: ComposeTransforms(
            [
                ForwardFillImputerTransform(),
                StdScalingTransform(dict_mean[var], dict_std[var]),
                FillWithZerosImputerTransform(),
                SamplingToReferenceTimeTransform(SETTINGS.BETA_VAE.ref_freq),
            ]
        )
        for var in [f"{source}-{signal}" for source, signal in source_signal_list]
    }

    # Shot-level transform for β-VAE
    shot_transform = ComposeTransforms(
        [
            TruncationTransform(),
            WindowSegmenterTransform(**PARAMETERS_WINDOWS_SEGMENTER),
            BetaVAETransform(),
        ]
    )

    # Prepare datasets
    datasets_train_val_test = initialize_datasets(
        sources_and_signals=source_signal_list,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=True,
    )

    # Prepare dataloaders
    dataloaders_train_val_test = initialize_dataloaders(
        datasets=datasets_train_val_test,
        collate_function=beta_vae_collate_fn,
        batch_size= SETTINGS.TRAINING.dataloader_batch_size,
        num_workers=SETTINGS.TRAINING.num_workers,
        verbose=True,
    )
    train_dataloader = dataloaders_train_val_test["train"]
    val_dataloader = dataloaders_train_val_test["val"]
    test_dataloader = dataloaders_train_val_test["test"]

    # Create β-VAE models
    beta_vae_models = create_beta_vae_models(
        train_dataloader,
        SETTINGS.BETA_VAE.latent_dim,
        SETTINGS.BETA_VAE.hidden_dim,
        SETTINGS.BETA_VAE.beta,
        verbose=True
    )

    best_model_states, training_loss_curves = train_beta_vae_models(
        beta_vae_models,
        train_dataloader,
        val_dataloader,
        OUTPUT_SUB_FOLDER,
        verbose=True,
        use_cosineLr=SETTINGS.BETA_VAE.use_cosine_lr
    )

    visualize_beta_vae_results(
        beta_vae_models,
        train_dataloader,
        val_dataloader,
        OUTPUT_SUB_FOLDER,
        training_loss_curves,
        verbose=True,
    )

    print("\n\n----------TRAINING COMPLETE----------")
    print(f"Trained β-VAE models for {len(best_model_states)} signals")
    print(f"Models saved in: output/{OUTPUT_SUB_FOLDER}")
