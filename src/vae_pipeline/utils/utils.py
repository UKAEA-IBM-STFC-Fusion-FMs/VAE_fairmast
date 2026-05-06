

def training_block(start, end, x, use_amp, beta, mu, logvar, clamp_logvar):
    x_sub_batch = x[start:end]

    # For a 3D signals (i.e., x_sub_batch dimension == 4) we use a conv2d encoder.
    # we must permute the indeces of our tensor to agree with the PyTorch conv2d convention.
    if x_sub_batch.ndim == 4:
        x_sub_batch = x_sub_batch.permute(0, 3, 1, 2).contiguous() 

    # Mask non-finite entries and NaN
    mask = torch.isfinite(x_sub_batch) # booleans
    mask = mask.float() # floats

    # Impute NaN with zeros, i.e., the mean of signals after standardization
    x_sub_batch = torch.nan_to_num(x_sub_batch, nan=0.0)
    x_mask_cat = torch.cat([x_sub_batch,mask], dim=1)

    try:
        with torch.amp.autocast('cuda', enabled=use_amp):
            x_recon, mu, logvar = model(x_mask_cat)
            loss, recon_loss, kl_loss = masked_loss_function(
                beta,
                x_recon[:,:x_sub_batch.shape[1]],
                x_sub_batch,
                mu,
                logvar,
                mask,
                clamp_logvar
            )
    except ValueError as e:
        # skip this sub-batch
        print(f"[batch {batch_idx} {start}:{end}] Error in loss calc: {e}")
        return None, None, None, None

    return loss, recon_loss, kl_loss, x_sub_batch.size(0)