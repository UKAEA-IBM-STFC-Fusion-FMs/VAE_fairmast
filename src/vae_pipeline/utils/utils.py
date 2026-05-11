import sys
import os

REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
import torch
from src.vae_pipeline.models.vae_model import masked_loss_function

def training_block(x_cat, model, use_amp, beta, clamp_logvar):
    
    """
    Run a single forward + loss-computation step for VAE training.

    Automatic Mixed Precision (AMP) is used if enabled.

    Parameters
    ----------
    x_cat : torch.Tensor
        Input tensor containing signal and concatenated mask.
    model : torch.nn.Module
        VAE model.
    use_amp : bool
        Whether to enable AMP autocasting during the forward pass.
    beta : float
        Weight of the KL divergence term in the VAE loss.
    clamp_logvar : bool
        Whether to clamp the log-variance for numerical stability.

    Returns
    -------
    loss : torch.Tensor or None
        Total VAE loss (reconstruction + beta * KL).
    recon_loss : torch.Tensor or None
        Masked reconstruction loss.
    kl_loss : torch.Tensor or None
        KL divergence term.
    x_recon : torch.Tensor or None
        Reconstructed signal.
    mu : torch.Tensor or None
        Latent mean.
    logvar : torch.Tensor or None
        Latent log-variance.
    mask : torch.Tensor or None
        Validity mask used for the masked loss.
    """

    if use_amp:
        with torch.amp.autocast('cuda', enabled=use_amp):
            try:
                breakpoint()
                x_recon, mu, logvar, mask = model(x_cat)
                loss, recon_loss, kl_loss = masked_loss_function(
                    beta,
                    x_recon,
                    x_cat,
                    mu,
                    logvar,
                    mask,
                    clamp_logvar
                )
            except ValueError as e:
                # skip this sub-batch
                print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
                return None, None, None, None, None, None, None
    else:
        try:
            x_recon, mu, logvar, mask = model(x_cat)
            loss, recon_loss, kl_loss = masked_loss_function(
                beta,
                x_recon,
                x_cat,
                mu,
                logvar,
                mask,
                clamp_logvar
            )
        except ValueError as e:
            # skip this sub-batch
            print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
            return None, None, None, None, None, None, None

    return  loss, recon_loss, kl_loss, x_recon, mu, logvar, mask