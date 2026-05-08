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

def training_block(x, model, use_amp, beta, clamp_logvar):

    # For a 3D signals (i.e., x dimension == 4) we use a conv2d encoder.
    # we must permute the indeces of our tensor to agree with the PyTorch conv2d convention.
    if x.ndim == 4:
        x = x.permute(0, 3, 1, 2).contiguous() 

    # Mask non-finite entries and NaN
    mask = torch.isfinite(x) # booleans
    mask = mask.float() # floats

    # Impute NaN with zeros, i.e., the mean of signals after standardization
    x = torch.nan_to_num(x, nan=0.0)
    x_mask_cat = torch.cat([x,mask], dim=1)

    if use_amp:
        with torch.amp.autocast('cuda', enabled=use_amp):
            try:
                x_recon, mu, logvar = model(x_mask_cat)
                loss, recon_loss, kl_loss = masked_loss_function(
                    beta,
                    x_recon[:,:x.shape[1]],
                    x,
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
            x_recon, mu, logvar = model(x_mask_cat)
            loss, recon_loss, kl_loss = masked_loss_function(
                beta,
                x_recon[:,:x.shape[1]],
                x,
                mu,
                logvar,
                mask,
                clamp_logvar
            )
        except ValueError as e:
            # skip this sub-batch
            print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
            return None, None, None, None, None, None, None

    return  loss, recon_loss, kl_loss, x_recon[:,:x.shape[1]], mu, logvar, mask