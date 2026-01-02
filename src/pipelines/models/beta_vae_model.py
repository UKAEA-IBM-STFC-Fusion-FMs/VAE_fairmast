import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """
    Beta-Variational Autoencoder for signal compression.
    
    Compresses 1D time series signals into a latent space representation
    with controllable disentanglement via the beta parameter.
    """

    def __init__(self, input_length, latent_dim=32, hidden_dim=128, beta=1.0):
        """
        Parameters
        ----------
        input_length : int
            Length of input time series
        latent_dim : int
            Dimensionality of latent space
        hidden_dim : int
            Hidden layer dimensions
        beta : float
            Beta parameter controlling KL divergence weight
        """
        super().__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = nn.Sequential(
            nn.Linear(input_length, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
        )
        
        # Latent space projection
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, input_length),
        )

    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation to reconstruction"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z

    def loss_function(self, reconstruction, target, mu, logvar):
        """
        β-VAE loss function
        
        Returns
        -------
        loss : torch.Tensor
            Total loss
        reconstruction_loss : torch.Tensor
            Reconstruction loss component
        kl_loss : torch.Tensor
            KL divergence loss component
        """
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    import os
    
    # Output directory for figures
    os.makedirs('output_figures', exist_ok=True)
    
    # Random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    input_length = 100
    latent_dim = 16
    hidden_dim = 64
    beta = 1.0
    n_samples = 1000
    batch_size = 32
    n_epochs = 250
    learning_rate = 1e-3
    
    print("Creating synthetic test data")

    # Mix of sine waves, noise, and step functions
    t = np.linspace(0, 4*np.pi, input_length)
    test_data = []
    
    for i in range(n_samples):
        freq1 = np.random.uniform(0.5, 3.0)
        freq2 = np.random.uniform(0.5, 3.0)
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        amp1 = np.random.uniform(0.5, 2.0)
        amp2 = np.random.uniform(0.5, 2.0)
        noise_level = np.random.uniform(0.1, 0.3)
        
        signal = (amp1 * np.sin(freq1 * t + phase1) + 
                 amp2 * np.cos(freq2 * t + phase2) + 
                 noise_level * np.random.randn(input_length))
        
        # Add some step functions occasionally
        if np.random.random() < 0.3:
            step_pos = np.random.randint(20, 80)
            step_height = np.random.uniform(-1, 1)
            signal[step_pos:] += step_height
            
        test_data.append(signal)
    
    test_data = torch.FloatTensor(np.array(test_data))
    
    # Normalize
    test_data = (test_data - test_data.mean()) / test_data.std()
    
    dataset = TensorDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Created {n_samples} synthetic signals of length {input_length}")
    
    # Initialize model
    model = BetaVAE(input_length, latent_dim, hidden_dim, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training Beta-VAE (β={beta}) for {n_epochs} epochs")
    
    # Training loop
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, (data,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            reconstruction, mu, logvar, z = model(data)
            
            loss, recon_loss, kl_loss = model.loss_function(reconstruction, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
                f"Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}")
    
    print("Training completed")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Total Loss')
    axes[0].plot(recon_losses, label='Reconstruction Loss')
    axes[0].plot(kl_losses, label='KL Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    model.eval()
    with torch.no_grad():
        # Select a few test samples
        test_indices = [0, 1, 2, 3, 4]
        test_samples = test_data[test_indices]
        
        # Get reconstructions
        reconstructions, mu, logvar, z = model(test_samples)
        
        # Plot original vs reconstructed
        for i, idx in enumerate(test_indices):
            offset = i * 3
            axes[1].plot(test_samples[i].numpy() + offset, 'b-', alpha=0.7, 
                    label='Original' if i == 0 else '')
            axes[1].plot(reconstructions[i].numpy() + offset, 'r--', alpha=0.7,
                    label='Reconstructed' if i == 0 else '')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Amplitude (offset)')
        axes[1].set_title('Original vs Reconstructed Signals')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot latent space (first 2 dimensions)
        # Get latent representations for more samples
        all_mu, _ = model.encode(test_data[:200])
        axes[2].scatter(all_mu[:, 0].numpy(), all_mu[:, 1].numpy(), alpha=0.6, s=20)
        axes[2].set_xlabel('Latent Dim 0')
        axes[2].set_ylabel('Latent Dim 1')
        axes[2].set_title('Latent Space Visualization (2D)')
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('output_figures/beta_vae_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model statistics
    print(f"\nModel Statistics:")
    print(f"Input Length: {input_length}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Compression Ratio: {input_length/latent_dim:.1f}:1")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Final Reconstruction Loss: {recon_losses[-1]:.4f}")
    print(f"Final KL Loss: {kl_losses[-1]:.4f}")
    
    # Test generation from random latent codes
    print("\nGenerating samples from random latent codes")
    with torch.no_grad():
        random_z = torch.randn(5, latent_dim)
        generated_samples = model.decode(random_z)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(5):
            ax.plot(generated_samples[i].numpy() + i * 2, label=f'Generated {i+1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Amplitude (offset)')
        ax.set_title('Generated Samples from Random Latent Codes')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('output_figures/beta_vae_generated_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Figures saved to output_figures/ directory")
