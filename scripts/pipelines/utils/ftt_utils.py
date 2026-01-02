"""
ft_transformer_utils
================================================================================
Utilities that support the FMfusion multimodal FT-Transformer training loop:

• Supervision building:
  Convert native targets (H,W,T-style tensors) into prediction-space targets
  (flatten or encoder-produced embeddings) to match each head’s output size.

• Data adapters & dataset:
  Helpers to convert list-structured raw targets to per-sample dicts and a
  minimal Dataset/Collate that preserves per-target batches without padding.

• Losses:
  Per-target balanced losses both in prediction space and in native space
  (the latter decodes predictions before scoring).

• Optimizer & scheduling:
  Build a persistent optimizer with one group for trunk and one group per head,
  plus toggling to only update heads present in the current batch.

• Training loop:
  A reference trainer that keeps one optimizer and toggles head groups per batch.

• Evaluation & visualization:
  Decode to native space for metrics, and utilities to visualize time series,
  per‑channel profiles, images, and videos.

• Synthetic data:
  Lightweight generators for quick end-to-end sanity checks.

Notes
-----
- Encoders expect float32; we handle dtype conversions where needed.
- Decoding requires per-batch metadata; pass Y_for_meta when requesting decoded outputs.
- For plotting, this module uses matplotlib
"""

from __future__ import annotations

import copy
import math
from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from contextlib import nullcontext
# import contextlib

from scripts.pipelines.models.ftt_model import InputRegistry, TargetRegistry, InputSpec, TargetSpec

# ======================================================================================================================
# Preliminaries

# ----------------------------------------------------------------------------------------------------------------------
# Choose AMP dtype (prefer bf16 on Ampere+) during training and evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16 
elif device.type == "cuda":
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = None

use_amp = (device.type == "cuda") and (AMP_DTYPE is not None)


# ----------------------------------------------------------------------------------------------------------------------
# Define supporting data types

Shape3D = Tuple[int, int, int]  # (d1, d2, d3)


# ----------------------------------------------------------------------------------------------------------------------
def amp_ctx():
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp)
    else:
        return nullcontext()


# ======================================================================================================================
# Supervision builder (native -> prediction space)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def build_supervision_from_native(
    y_native: Dict[str, torch.Tensor],
    registry,  # model.registry
) -> Dict[str, torch.Tensor]:
    """
    Convert native targets to the *prediction space* (what each head outputs).

    For each target:
      - If a decoder exists, use the paired *encoder* to produce coefficients ('embedding').
      - Else, flatten the native tensor.

    Returns
    -------
    dict[name] -> (B, out_dim) torch.Tensor
    """

    y_flat: Dict[str, torch.Tensor] = {}
    for name, Yt in y_native.items():
        B = Yt.shape[0]
        dec = registry.decoders.get(name)
        if dec is None:
            y_flat[name] = Yt.reshape(B, -1).to(Yt.dtype)
        else:
            enc = registry.encoders[name]
            coeffs = []
            for b in range(B):
                meta_b = enc.encode(Yt[b].to(dtype=torch.float32))  # encoders expect fp32
                emb_b = meta_b["embedding"].reshape(-1)
                coeffs.append(emb_b)
            y_flat[name] = torch.stack(coeffs, dim=0).to(Yt.dtype)
    return y_flat


# ======================================================================================================================
# Adapters: list -> dict (+ active target names)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def yraw_list_to_dicts(
    Y_raw: List[List[np.ndarray]],
    target_order: List[str],
    *,
    drop_missing: bool = True,
) -> tuple[list[dict[str, np.ndarray]], list[list[str]]]:
    """
    Transform list-structured targets into per-sample dicts (and record active names).

    Parameters
    ----------
    Y_raw : List over samples; each sample is a List aligned to `target_order`.
            Each entry is a native np.ndarray (d1,d2,d3) or None if absent.
    target_order : The canonical order of targets.
    drop_missing : If True, skip None entries. (If False, None is still skipped but call sites can validate presence
                   separately.)

    Returns
    -------
    Y_dicts        : List[Dict[target_name, np.ndarray]]
    active_targets : List[List[target_name]] (names present for each sample, order preserved)
    """

    Y_dicts: list[dict[str, np.ndarray]] = []
    active_targets: list[list[str]] = []

    for sample in Y_raw:
        assert len(sample) == len(target_order), "Y_raw sample length != target_order length"
        ydict: dict[str, np.ndarray] = {}
        names: list[str] = []

        for name, arr in zip(target_order, sample):
            if arr is None:
                if drop_missing:
                    continue
                continue  # keep behavior identical; absent targets are skipped
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Y_raw contains non-numpy entry for target '{name}'")
            ydict[name] = arr
            names.append(name)

        Y_dicts.append(ydict)
        active_targets.append(names)

    return Y_dicts, active_targets


# ======================================================================================================================
# Loss (per-target, device-safe)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def compute_per_target_loss_pred_space(
    preds: Dict[str, torch.Tensor],           # name -> (B, D)
    targets_flat: Dict[str, torch.Tensor],    # name -> (B, D)
    registry,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Per-target MSE in prediction space with per-sample reduction:
      1) mean over feature dim per sample
      2) mean over samples
      3) weighted across targets (weights normalized to sum=1 when > 0)
    """

    per_t = []
    names = []
    logs: Dict[str, float] = {}

    for name, y_true in targets_flat.items():
        if name not in preds:
            continue
        y_pred = preds[name]
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)

        # --- Per-sample mean (shapes are the same in the same targets)
        # Shapes: yp, yt = (B, d1, d2, d3)
        l_ = torch.mean((y_pred - y_true) ** 2)

        # Version for weighting different each samples and introducing masking at a sample level
        # diff2 = (yp - yt) ** 2
        # per_sample = diff2.flatten(1).mean(dim=1)   # (B,)
        # l_ = per_sample.mean()

        per_t.append(l_)
        names.append(name)
        logs[name] = float(l_.detach().cpu())

    if not per_t:
        # Nothing active in this batch
        # Safer device/dtype fallback if preds could be empty
        if len(preds) == 0:
            return torch.tensor(0.0), logs
        any_tensor = next(iter(preds.values()))
        return torch.zeros((), device=any_tensor.device, dtype=any_tensor.dtype), logs

    per_t = torch.stack(per_t, dim=0)  # (T_active,)
    weights = torch.tensor(
        [registry.specs[n].loss_weight for n in names],
        device=per_t.device,
        dtype=per_t.dtype
    )
    if float(weights.sum()) > 0:
        weights = weights / (weights.sum() + 1e-8)
        loss = (per_t * weights).sum()
    else:
        loss = per_t.mean()
    return loss, logs


# ----------------------------------------------------------------------------------------------------------------------
def decode_preds_to_native(
    model,
    preds: Dict[str, torch.Tensor],
    y_native_for_meta: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Decode per-target preds back to native shapes (differentiable)."""

    meta_batch = model.encode_output_metadata(y_native_for_meta)
    return model.decode_per_target(preds, meta_batch, model.registry.specs)


# ----------------------------------------------------------------------------------------------------------------------
def compute_per_target_loss_native_space(
    model,
    preds: Dict[str, torch.Tensor],
    y_native: Dict[str, torch.Tensor],
    registry,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Per-target MSE in *native space* with per-sample reduction, then weighted across targets. Returns a scalar loss for
    backward and per-target logs (means).
    """

    y_pred_native = decode_preds_to_native(model, preds, y_native)

    per_t = []
    names = []
    logs: Dict[str, float] = {}
    _printed_diag = getattr(compute_per_target_loss_native_space, "_printed_diag", False)

    for name, yt in y_native.items():
        if name not in y_pred_native:
            continue
        yp = y_pred_native[name]
        yt = yt.to(device=yp.device, dtype=yp.dtype)
        if (not _printed_diag) and (not yp.requires_grad):
            print(f"[warn] decoded '{name}' does not require grad. "
                  f"Check for @torch.no_grad or .detach() in decode path.")

        # --- per-sample mean (shapes are the same in the same targets)
        # shapes: yp, yt = (B, d1, d2, d3)
        l_ = torch.mean((yp - yt) ** 2)

        # version for weighting different each samples and introducing masking at a sample level
        # diff2 = (yp - yt) ** 2
        # per_sample = diff2.flatten(1).mean(dim=1)   # (B,)
        # l_ = per_sample.mean()

        per_t.append(l_)
        names.append(name)
        logs[name] = float(l_.detach().cpu())

    setattr(compute_per_target_loss_native_space, "_printed_diag", True)

    if not per_t:
        any_tensor = next(iter(preds.values()))
        return torch.zeros((), device=any_tensor.device, dtype=any_tensor.dtype), logs

    per_t = torch.stack(per_t, dim=0)  # (T,)
    weights = torch.tensor(
        [registry.specs[n].loss_weight for n in names],
        device=per_t.device,
        dtype=per_t.dtype
    )
    loss = (per_t * (weights / (weights.sum() + 1e-8))).sum() if float(weights.sum()) > 0 else per_t.mean()

    return loss, logs


# ======================================================================================================================
# Optimizer: per-head groups + toggling
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def build_param_groups(
    model: nn.Module,
    *,
    lr_trunk: float,
    wd_trunk: float,
    lr_heads: float,
    wd_heads: float,
    include_decoders: bool = False,
):
    """
    Build one param group for the trunk (always active) and one per head (start inactive).
    Optionally add a group per decoder if decoders are trainable.
    """

    # Trunk group (always active); store base_lr to compute scheduler scaling later.
    groups = [
        {
            "params": list(model.backbone.parameters()),
            "lr": lr_trunk,
            "weight_decay": wd_trunk,
            "group_type": "trunk",
            "base_lr": lr_trunk,
        }
    ]

    # Heads start disabled (lr=0, wd=0). We store desired "on" values.
    for name, head in model.heads.items():
        groups.append({
            "params": list(head.parameters()),
            "lr": 0.0,
            "weight_decay": 0.0,
            "group_type": "head",
            "head_name": name,
            "lr_on": lr_heads,
            "wd_on": wd_heads,
        })

        # If your decoders are nn.Module and you want them trainable, include them here.
        if include_decoders:
            dec = model.decoders.get(name)
            if isinstance(dec, nn.Module):
                groups.append({
                    "params": list(dec.parameters()),
                    "lr": 0.0,
                    "weight_decay": 0.0,
                    "group_type": "decoder",
                    "head_name": name,
                    "lr_on": lr_heads,
                    "wd_on": wd_heads,
                })

    return groups


# ----------------------------------------------------------------------------------------------------------------------
def _trunk_lr_scale(optimizer: torch.optim.Optimizer) -> float:
    """
    Read scheduler-scaled LR from the trunk group to derive a scale factor applied to heads.
    scale = current_trunk_lr / base_trunk_lr
    """

    for g in optimizer.param_groups:
        if g.get("group_type") == "trunk":
            base = float(g.get("base_lr", g["lr"]))
            curr = float(g["lr"])
            return 1.0 if base <= 0 else curr / base

    return 1.0


# ----------------------------------------------------------------------------------------------------------------------
def toggle_head_groups(optimizer: torch.optim.Optimizer, active_names: set[str]):
    """
    Enable LR/WD for active heads; disable for others.
    Ensures inactive heads do not update and are not decayed by AdamW.
    Also matches scheduler scale (by trunk group) so heads track the trunk LR schedule.
    """

    scale = _trunk_lr_scale(optimizer)
    for g in optimizer.param_groups:
        gt = g.get("group_type", "")
        if gt in ("head", "decoder"):
            if g.get("head_name") in active_names:
                g["lr"] = float(g.get("lr_on", 0.0)) * scale
                g["weight_decay"] = float(g.get("wd_on", 0.0))
            else:
                g["lr"] = 0.0
                g["weight_decay"] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_trunk: float = 1e-3,
    lr_heads: float = 2e-3,
    wd_trunk: float = 1e-4,
    wd_heads: float = 0.0,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    use_adamw: bool = True,
    include_decoders: bool = False,
):
    """
    Create a persistent optimizer over trunk + ALL heads (each head its own group).
    Inactive heads are kept with lr=0 and weight_decay=0 and won't update or decay.
    """

    param_groups = build_param_groups(
        model,
        lr_trunk=lr_trunk, wd_trunk=wd_trunk,
        lr_heads=lr_heads, wd_heads=wd_heads,
        include_decoders=include_decoders,
    )

    Optim = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = Optim(param_groups, betas=(0.9, 0.999), eps=1e-8)

    scheduler = None
    if total_steps is not None and total_steps > 0:
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, step / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


# ======================================================================================================================
# Trainer (persistent opt; per-batch head toggling)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def train_model_per_target_persistent(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    registry,
    epochs: int = 100,
    lr_trunk: float = 1e-3,
    lr_heads: float = 2e-3,
    wd_trunk: float = 1e-4,
    wd_heads: float = 0.0,
    warmup_steps: int = 0,
    use_adamw: bool = True,
    grad_accum_steps: int = 1,
    patience: int = 5,
    min_delta: float = 0.0,
    restore_best: bool = True,
    loss_space: str = "pred",  # 'pred' or 'native'
    verbose: bool = True,
):
    """
    Reference trainer with AMP:
      - Model stays fp32; compute is bf16/fp16 under autocast on CUDA.
      - One persistent optimizer across trunk + all heads.
      - Toggle head parameter groups per batch based on active targets.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Set Up scaler for AMP

    scaler = torch.amp.GradScaler(enabled=use_amp and AMP_DTYPE == torch.float16)

    # ------------------------------------------------------------------------------------------------------------------
    # Optimizer/Scheduler

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch // max(1, grad_accum_steps)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_trunk=lr_trunk,
        lr_heads=lr_heads,
        wd_trunk=wd_trunk,
        wd_heads=wd_heads,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        use_adamw=use_adamw,
        include_decoders=False,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Pretty print

    if verbose:

        try:
            _xb, _names, _yn = next(iter(train_loader))
            active_targets_print = list(_names)
        except Exception as ee:
            print(f"Unexpected Exception while calculating `active_targets_print`: {ee}")
            active_targets_print = list(model.heads.keys())

        n_trunk = sum(p.numel() for p in model.backbone.parameters())
        n_heads = sum(p.numel() for h in model.heads.values() for p in h.parameters())
        n_total = sum(p.numel() for p in model.parameters())

        amp_str = "off"
        if use_amp:
            amp_str = f"on ({'bf16' if AMP_DTYPE==torch.bfloat16 else 'fp16'+', GradScaler'})"

        print("\n[train] Configuration")
        print(f"  device={device} | AMP={amp_str}")
        print(f"  epochs={epochs} | batches/epoch={len(train_loader)} | total_steps≈{total_steps}")
        print(f"  optimizer={'AdamW' if use_adamw else 'Adam'} | lr_trunk={lr_trunk} lr_heads={lr_heads} | "
              f"wd_trunk={wd_trunk} wd_heads={wd_heads}")
        print(f"  scheduler={'cosine+warmup' if (total_steps > 0) else 'None'} | warmup_steps={warmup_steps}")
        print(f"  grad_accum_steps={grad_accum_steps}")
        print(f"  early_stopping: patience={patience} min_delta={min_delta} restore_best={restore_best}")
        print(f"  loss_space={loss_space}  (balanced per target; weights from TargetSpec.loss_weight)")
        print(f"  params: trunk={n_trunk/1e6:.2f}M heads={n_heads/1e6:.2f}M total={n_total/1e6:.2f}M")
        print(f"  available targets: {list(model.heads.keys())}")
        print(f"  active targets: {active_targets_print}\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Training loop

    history = {"train_loss": [], "val_loss": []}
    best_val = math.inf
    best_state = None
    bad_epochs = 0

    model.train()
    for epoch in range(1, epochs + 1):
        train_loss_sum = 0.0
        train_sample_count = 0

        optimizer.zero_grad(set_to_none=True)
        last_active_t = None
        it = 0

        for it, (X_batch, active_t_batch, y_native) in enumerate(train_loader, start=1):
            last_active_t = active_t_batch

            with amp_ctx():
                out = model(
                    X_batch,
                    active_targets=active_t_batch,
                    Y_for_meta=y_native if (loss_space == "native") else None,
                    require_decoded=False,
                )
                preds = out["preds"]

                if loss_space == "native":
                    loss, _ = compute_per_target_loss_native_space(model, preds, y_native, registry)
                else:
                    y_flat = build_supervision_from_native(y_native, registry)
                    loss, _ = compute_per_target_loss_pred_space(preds, y_flat, registry)

                # Keep scalar in fp32 to avoid overflow
                loss = loss.float()

            # Sample-weighted running mean
            any_t = next(iter(y_native.values()), None)
            if any_t is None:
                continue

            B = int(any_t.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * B
            train_sample_count += B

            if scaler.is_enabled():
                scaler.scale(loss / max(1, grad_accum_steps)).backward()
            else:
                (loss / max(1, grad_accum_steps)).backward()

            if it % grad_accum_steps == 0:
                toggle_head_groups(optimizer, active_names=set(active_t_batch))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                    toggle_head_groups(optimizer, active_names=set(active_t_batch))

        # Finalize partial accumulation at epoch end
        if (len(train_loader) > 0) and ((it % max(1, grad_accum_steps)) != 0) and (last_active_t is not None):
            toggle_head_groups(optimizer, active_names=set(last_active_t))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
                toggle_head_groups(optimizer, active_names=set(last_active_t))

        train_loss = train_loss_sum / max(1, train_sample_count)
        history["train_loss"].append(train_loss)

        # ..............................................................................................................
        # Validation (no grad, same AMP)

        model.eval()
        val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for X_batch, active_t_batch, y_native in val_loader:
                with amp_ctx():
                    out = model(
                        X_batch,
                        active_targets=active_t_batch,
                        Y_for_meta=y_native if (loss_space == "native") else None,
                        require_decoded=False,
                    )
                    preds = out["preds"]

                    if loss_space == "native":
                        vloss, _ = compute_per_target_loss_native_space(model, preds, y_native, registry)
                    else:
                        y_flat = build_supervision_from_native(y_native, registry)
                        vloss, _ = compute_per_target_loss_pred_space(preds, y_flat, registry)

                    vloss = vloss.float()

                any_t = next(iter(y_native.values()))
                B = int(any_t.shape[0])
                val_loss_sum += float(vloss.detach().cpu()) * B
                val_sample_count += B

        val_loss = val_loss_sum / max(1, val_sample_count)
        history["val_loss"].append(val_loss)

        improved = (val_loss < best_val - float(min_delta))
        tag = "← best" if improved else f"({bad_epochs}/{patience})"
        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} {tag}")

        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_val:.6f}).")
                break

        model.train()

    if restore_best and (best_state is not None):
        model.load_state_dict(best_state)

    return history


# ======================================================================================================================
# Evaluation (decode to native + visualize bridge)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_padded_from_dicts(
    y_native: Dict[str, torch.Tensor],
    y_pred_native: Dict[str, torch.Tensor],
    target_names: List[str],
    padded_output_shape: Shape3D,
):
    """
    Create fixed-size (B, T, D1, D2, D3) tensors by zero-padding each per-target tensor.
    Useful for grid visualization across heterogeneous target shapes.
    """

    assert all(n in y_native for n in target_names), "Missing target in y_native"
    B = next(iter(y_native.values())).shape[0]
    T = len(target_names)
    D1, D2, D3 = padded_output_shape

    # allocate on CPU; we'll return numpy
    Y_true_pad = torch.zeros((B, T, D1, D2, D3), dtype=torch.float32, device="cpu")
    Y_pred_pad = torch.zeros_like(Y_true_pad)

    for t, name in enumerate(target_names):
        yt = y_native[name].to(dtype=torch.float32, device="cpu")
        yp = y_pred_native[name].to(dtype=torch.float32, device="cpu")
        d1, d2, d3 = yt.shape[1:4]
        Y_true_pad[:, t, :d1, :d2, :d3] = yt
        Y_pred_pad[:, t, :d1, :d2, :d3] = yp

    return Y_true_pad.numpy(), Y_pred_pad.numpy()


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model_per_target(
    model,
    test_loader,
    *,
    target_names: List[str],
    padded_output_shape: Shape3D,
):
    """
    Evaluate one batch from `test_loader` (bucketed to a single target order) and return padded arrays for visualization
    along with RMSE metrics.
    """

    # ..................................................................................................................
    # Overall RMSE in prediction space (concatenate flats) — PER-SAMPLE semantics
    def _cat(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([d[n_].reshape(d[n_].shape[0], -1) for n_ in active_targets], dim=1)

    # ..................................................................................................................

    model.eval()
    X_batch, active_targets, y_native = next(iter(test_loader))
    assert active_targets == target_names, "For plotting, pass loader bucketed to the same target order."

    with amp_ctx():
        out = model(
            X_batch,
            active_targets=active_targets,
            Y_for_meta=y_native,
            require_decoded=False,
        )
        preds = out["preds"]
    y_pred_native = decode_preds_to_native(model, preds, y_native)

    # Align device & dtype before math
    y_flat_pred = _cat(preds).float()
    y_flat_true = _cat(build_supervision_from_native(y_native, model.registry))
    y_flat_true = y_flat_true.to(device=y_flat_pred.device, dtype=y_flat_pred.dtype)

    # Per-sample MSE -> mean over samples -> RMSE
    diff2 = (y_flat_pred - y_flat_true) ** 2            # (B, Dtot)
    per_sample_mse = diff2.mean(dim=1)                  # (B,)
    rmse_total = float(torch.sqrt(per_sample_mse.mean()).item())

    # Per-target RMSE (native) — PER-SAMPLE semantics
    pred_device = next(iter(y_pred_native.values())).device
    rmse_per_target = []
    for n in active_targets:
        yt = y_native[n].to(device=pred_device, dtype=torch.float32)
        yp = y_pred_native[n].to(device=pred_device, dtype=torch.float32)
        rmse_n = torch.sqrt(((yp - yt) ** 2).mean()).item()
        rmse_per_target.append(float(rmse_n))

        # Version for weighting different each samples and introducing masking at a sample level
        # diff2 = (yp - yt) ** 2                          # (B, d1, d2, d3)
        # per_sample_mse = diff2.reshape(diff2.shape[0], -1).mean(dim=1)  # (B,)
        # rmse_n = torch.sqrt(per_sample_mse.mean()).item()
        # rmse_per_target.append(float(rmse_n))

    Y_true_np, Y_pred_np = build_padded_from_dicts(
        y_native, y_pred_native, target_names=active_targets, padded_output_shape=padded_output_shape
    )

    print(f"\n✅ Test RMSE (flat space): {rmse_total:.4f}")
    print(f"✅ RMSE per target (native): {[round(x, 4) for x in rmse_per_target]}")

    return Y_true_np, Y_pred_np, rmse_total, rmse_per_target


# ======================================================================================================================
# Shape inference (still handy for plotting)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def infer_modality_from_shape(shape: tuple[int, ...]) -> str:
    """
    Infer a modality label from a tensor shape.
    Supports 1D, 2D (common for inputs), and canonical 3D (C,H,T) targets.

    Returns one of:
      'scalar', 'vector', 'timeseries', 'profile', 'image', 'video', 'unknown'
    """

    ndim = len(shape)

    # ..................................................................................................................
    # 1D

    if ndim == 1:
        c, = shape
        return "scalar" if c == 1 else "vector"

    # ..................................................................................................................
    # 2D (typical input: (C, T))

    if ndim == 2:
        d1, d2 = shape

        # Heuristic: treat as multichannel 1D signal over time if the second dim > 1
        if d2 > 1:
            return "timeseries"   # (C, T)

        # Otherwise it's a static vector-like thing
        return "vector"           # (C, 1) or (1, 1)

    # ..................................................................................................................
    # 3D (canonical targets: (C, H, T))

    if ndim == 3:
        d1, d2, d3 = shape

        # Pure scalar / vector (no time, no spatial width)
        if d2 == 1 and d3 == 1:
            return "scalar" if d1 == 1 else "vector"

        # Image or video (true 2D spatial)
        if d1 > 1 and d2 > 1:
            return "video" if d3 > 1 else "image"

        # 1D over time: decide "timeseries" vs "profile" by the first dim
        # (if there's only 1 channel → timeseries; if many positions → profile)
        if d2 == 1 and d3 > 1:
            return "timeseries" if d1 == 1 else "profile"

    # ..................................................................................................................

    return "unknown"


# ----------------------------------------------------------------------------------------------------------------------
def build_registries_from_shapes(
    *,
    input_names: List[str],
    input_shapes: List[Tuple[int, ...]],
    target_names: List[str],
    target_shapes: List[Tuple[int, ...]],
    get_encoder,  # name -> encoder factory
    get_decoder,  # name -> decoder factory
    input_overrides: Optional[Dict[str, Dict]] = None,
    target_overrides: Optional[Dict[str, Dict]] = None,
    default_input_by_modality: Optional[Dict[str, Dict]] = None,
    default_target_by_modality: Optional[Dict[str, Dict]] = None,
) -> Tuple[InputRegistry, TargetRegistry]:

    # --- basic checks ---
    assert len(input_names) == len(input_shapes), "input_names vs input_shapes mismatch"
    assert len(target_names) == len(target_shapes), "target_names vs target_shapes mismatch"
    assert len(set(input_names)) == len(input_names), "Duplicate input names"
    assert len(set(target_names)) == len(target_names), "Duplicate target names"

    input_overrides = input_overrides or {}
    target_overrides = target_overrides or {}

    # ..................................................................................................................
    # Sensible internal defaults if none provided

    if default_input_by_modality is None:
        default_input_by_modality = {
            "timeseries": dict(
                encoder_name="flatten_bspline_1d",
                encoder_kwargs={"degree": 4, "num_basis": 5}
            ),
            "profile": dict(
                encoder_name="fpca_3d",
                encoder_kwargs={"num_components": 3, "pca_dim": "space"}
            ),
            "image": dict(
                encoder_name="dct_2d",
                encoder_kwargs={"keep_h": 8, "keep_w": 8}),
            "video": dict(
                encoder_name="dct_3d",
                encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}
            ),
            "scalar": dict(
                encoder_name=None,
                encoder_kwargs=None
            ),
            "vector": dict(
                encoder_name=None,
                encoder_kwargs=None
            ),
        }

    if default_target_by_modality is None:
        default_target_by_modality = {
            "timeseries": dict(
                encoder_name="per_channel_bspline_1d",
                encoder_kwargs={"degree": 4, "num_basis": 5}
            ),
            "profile": dict(
                encoder_name="per_channel_bspline_1d",
                encoder_kwargs={"degree": 4, "num_basis": 5}
            ),
            "image": dict(
                encoder_name="dct_2d",
                encoder_kwargs={"keep_h": 8, "keep_w": 8}
            ),
            "video": dict(
                encoder_name="dct_3d",
                encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}
            ),
            "scalar": dict(
                encoder_name=None,
                encoder_kwargs=None
            ),
            "vector": dict(
                encoder_name=None,
                encoder_kwargs=None
            ),
        }

    # ..................................................................................................................
    # INPUT SPECS (objects)

    in_specs: Dict[str, InputSpec] = {}
    for name, shp in zip(input_names, input_shapes):
        mod = infer_modality_from_shape(shp)
        if mod == "unknown":
            raise ValueError(f"Unknown modality for input '{name}' with shape {shp}")

        enc_cfg = (input_overrides.get(name) or {}).copy()
        if enc_cfg.get("encoder_name") is None:
            enc_cfg.update(default_input_by_modality.get(mod, {}))

        in_specs[name] = InputSpec(
            name=name,
            shape=tuple(int(s) for s in shp),  # noqa
            encoder_name=enc_cfg.get("encoder_name"),
            encoder_kwargs=enc_cfg.get("encoder_kwargs"),
        )

    input_registry = InputRegistry(
        specs=in_specs,
        get_encoder=get_encoder,
        infer_modality_from_shape_fn=infer_modality_from_shape
    )
    input_registry.bind_shapes({n: s for n, s in zip(input_names, input_shapes)})
    input_registry.auto_fill_modalities()  # ok if your registry populates spec.modality internally
    input_registry.build_encoders()

    # ..................................................................................................................
    # TARGET SPECS (objects)

    tgt_specs: Dict[str, TargetSpec] = {}
    for name, shp in zip(target_names, target_shapes):
        mod = infer_modality_from_shape(shp)
        if mod == "unknown":
            raise ValueError(f"Unknown modality for target '{name}' with shape {shp}")

        enc_cfg = (target_overrides.get(name) or {}).copy()
        if enc_cfg.get("encoder_name") is None:
            enc_cfg.update(default_target_by_modality.get(mod, {}))

        tgt_specs[name] = TargetSpec(
            name=name,
            shape=tuple(int(s) for s in shp),  # noqa
            encoder_name=enc_cfg.get("encoder_name"),
            encoder_kwargs=enc_cfg.get("encoder_kwargs"),
            # head_hidden / loss / loss_weight can be left to defaults or set here if needed
        )

    target_registry = TargetRegistry(
        specs=tgt_specs,
        get_encoder=get_encoder,
        get_decoder=get_decoder,
    )
    target_registry.bind_shapes({n: s for n, s in zip(target_names, target_shapes)})
    target_registry.auto_fill_decoders()

    # ..................................................................................................................

    return input_registry, target_registry


# ======================================================================================================================
# Visualization
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def visualize_sample_outputs(
    Y_true,
    Y_pred,
    target_shapes,
    sample_indices=(0,),
    target_names=None,
    max_channels=20,
    frames_per_row=6,
):
    """
    Visualize decoded predictions vs ground truth across modalities.

    Parameters
    ----------
    Y_true : shape (B, T, D1_max, D2_max, D3_max)
    Y_pred : shape (B, T, D1_max, D2_max, D3_max)
    target_shapes : list[(d1,d2,d3)] aligned with T
    sample_indices :
    target_names :
    max_channels : Used only for (C,1,T) plots
    frames_per_row : How many frames per row for videos (wraps if needed)
    """

    B, T = Y_true.shape[0], Y_true.shape[1]
    assert Y_pred.shape[:2] == (B, T), "Y_true and Y_pred batch/target dims must match"
    assert len(target_shapes) == T, "target_shapes must align with T"

    for i in sample_indices:
        assert 0 <= i < B, f"sample index {i} out of range [0,{B})"
        for t, (d1, d2, d3) in enumerate(target_shapes):
            yt = Y_true[i, t, :d1, :d2, :d3]
            yp = Y_pred[i, t, :d1, :d2, :d3]
            tname = target_names[t] if (target_names and t < len(target_names)) else f"Target {t}"

            # ..........................................................................................................
            # Time series (1, 1, T)

            if d1 == 1 and d2 == 1 and d3 > 1:
                plt.figure()
                plt.plot(yt[0, 0, :], label="True")
                plt.plot(yp[0, 0, :], label="Pred")
                plt.title(f"Sample {i}, {tname} (Time Series)")
                plt.legend()
                plt.tight_layout()
                plt.show()

            # ..........................................................................................................
            # Per-channel profiles (C, 1, T)

            elif d1 > 1 and d2 == 1 and d3 > 1:
                C = d1
                num = min(C, max_channels)
                fig, axs = plt.subplots(num, 1, figsize=(8, 2 * num), sharex=True)
                axs = np.atleast_1d(axs)
                for c in range(num):
                    axs[c].plot(yt[c, 0, :], label="True")  # noqa
                    axs[c].plot(yp[c, 0, :], label="Pred")  # noqa
                    axs[c].set_ylabel(f"Ch {c}")  # noqa
                    axs[c].grid(True, alpha=0.2)  # noqa
                    axs[c].legend()  # noqa
                if C > num:
                    fig.suptitle(f"Sample {i}, {tname} (2D Profile) — first {num}/{C} channels")
                else:
                    fig.suptitle(f"Sample {i}, {tname} (2D Profile)")
                plt.tight_layout()
                plt.show()

            # ..........................................................................................................
            # Image/video (H, W, T) -> Show ALL frames wrapped

            elif d1 > 1 and d2 > 1 and d3 >= 1:
                vmin = float(min(yt.min(), yp.min()))
                vmax = float(max(yt.max(), yp.max()))

                for start in range(0, d3, frames_per_row):
                    end = min(start + frames_per_row, d3)
                    ncols = end - start
                    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 5))
                    if ncols == 1:
                        axes = np.array([[axes[0]], [axes[1]]])  # Normalize shape

                    for k, fidx in enumerate(range(start, end)):
                        axes[0, k].imshow(yt[:, :, fidx], cmap="viridis", vmin=vmin, vmax=vmax)
                        axes[0, k].set_title(f"True f{fidx}")
                        axes[0, k].axis("off")

                        axes[1, k].imshow(yp[:, :, fidx], cmap="viridis", vmin=vmin, vmax=vmax)
                        axes[1, k].set_title(f"Pred f{fidx}")
                        axes[1, k].axis("off")

                    kind = "Image" if d3 == 1 else "Video"
                    fig.suptitle(f"Sample {i}, {tname} ({kind}) — frames {start}..{end-1}")
                    plt.tight_layout()
                    plt.show()

            # ..........................................................................................................
            # Vector (D,1,1)

            elif d1 >= 1 and d2 == 1 and d3 == 1:
                vals_true = yt[:, 0, 0]
                vals_pred = yp[:, 0, 0]
                idx = np.arange(d1)
                plt.figure(figsize=(7, max(3, d1 * 0.15)))
                plt.scatter(vals_true, idx, label="True", marker="o")  # noqa
                plt.scatter(vals_pred, idx, label="Pred", marker="x")  # noqa
                for k in range(d1):
                    plt.plot([vals_true[k], vals_pred[k]], [idx[k], idx[k]],  # noqa
                             linewidth=0.8, alpha=0.5, color="grey", zorder=0)
                plt.title(f"Sample {i}, {tname} (Vector)")
                plt.xlabel("Value")
                plt.ylabel("Index")
                plt.gca().invert_yaxis()
                plt.legend()
                plt.tight_layout()
                plt.show()

            else:
                print(f"[visualize] Unhandled shape for {tname}: (d1={d1}, d2={d2}, d3={d3})")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Below classes and functions are designed for synthetic data experiments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ======================================================================================================================
# Dataset / Collate (per-target dicts; no pad/mask)
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def infer_shapes_from_raw(
    X_raw: List[List[np.ndarray]],
    Y_raw: List[List[np.ndarray]],
) -> Dict[str, object]:
    """
    This is for synthetic data experiments.
    Infer input/target shapes & modalities from the first sample and validate consistency.
    Also compute a padded_output_shape that can hold all targets for grid visualization.
    """

    input_shapes = [tuple(arr.shape) for arr in X_raw[0]]
    target_shapes = [tuple(arr.shape) for arr in Y_raw[0]]
    for Xi in X_raw:
        assert [tuple(a.shape) for a in Xi] == input_shapes, "Inconsistent input shapes across samples."
    for Yi in Y_raw:
        assert [tuple(a.shape) for a in Yi] == target_shapes, "Inconsistent target shapes across samples."
    input_modalities = [infer_modality_from_shape(s) for s in input_shapes]
    output_modalities = [infer_modality_from_shape(s) for s in target_shapes]
    max_d1 = max(s[0] for s in target_shapes) if target_shapes else 0
    max_d2 = max(s[1] for s in target_shapes) if target_shapes else 0
    max_d3 = max(s[2] for s in target_shapes) if target_shapes else 0
    padded_output_shape = (max_d1, max_d2, max_d3)

    return dict(
        input_shapes=input_shapes,
        target_shapes=target_shapes,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        padded_output_shape=padded_output_shape,
    )


# ======================================================================================================================
class PerTargetDataset(Dataset):
    """
    Minimal dataset that carries raw inputs and per-target native arrays.

    X_raw[b]         : List[n_inputs] of np.ndarray (native)
    Y_raw_per_target : List over samples of Dict[target_name] -> np.ndarray(native (d1,d2,d3))
    active_targets[b]: List[str] names present in Y_raw[b]
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        X_raw,
        Y_raw_per_target: List[Dict[str, np.ndarray]],
        active_targets: List[List[str]],
        dtype: torch.dtype = torch.float32
    ):

        assert len(X_raw) == len(Y_raw_per_target) == len(active_targets), "Mismatched dataset lengths."
        self.X_raw = X_raw
        self.Y_raw = Y_raw_per_target
        self.active_targets = active_targets
        self.dtype = dtype

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.X_raw)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        Xs = self.X_raw[idx]  # List of np arrays (kept as-is; model handles conversion)
        names = self.active_targets[idx]
        y_native = {n: torch.from_numpy(self.Y_raw[idx][n].astype(np.float32)) for n in names}

        return Xs, names, y_native


# ----------------------------------------------------------------------------------------------------------------------
def collate_per_target(batch, dtype: torch.dtype = torch.float32):
    """
    Collate samples that share the same active_targets (recommend: bucket by task).

    Returns
    -------
    X_batch       : List[B] of List[n_inputs] (raw np arrays)
    active_targets: List[str] (shared for the batch)
    y_native      : Dict[name] -> (B, d1, d2, d3) tensor
    """

    if not batch:
        return [], [], {}
    X_batch = [b[0] for b in batch]
    active_targets = batch[0][1]
    for _, names, _ in batch:
        if names != active_targets:
            raise ValueError("Batch mixes different active_targets; bucket by task before batching.")

    # Stack native Ys per target
    stacked: Dict[str, List[torch.Tensor]] = {n: [] for n in active_targets}
    for _, _, y_nat in batch:
        for n in active_targets:
            stacked[n].append(y_nat[n])
    y_native = {n: torch.stack(tlist, dim=0).to(dtype) for n, tlist in stacked.items()}

    return X_batch, active_targets, y_native


# ======================================================================================================================
# Synthetic data generation (kept, lightly annotated) — great for testing new models
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def sample_latents(rng: np.random.Generator) -> Dict[str, Any]:
    """Shared latent variables coupling X and Y."""

    z = {
        "amp": rng.uniform(0.6, 1.4),
        "phase": rng.uniform(0, 2*np.pi),
        "freq": rng.uniform(0.8, 1.6),
        "bias": rng.normal(0, 0.1),
        "xy_shift": rng.uniform(-0.15, 0.15, 2),
        "temporal_shift": rng.uniform(-0.2, 0.2),
        "contrast": rng.uniform(0.8, 1.3),
        "texture": rng.uniform(0.0, 0.25),
    }

    return z


# ----------------------------------------------------------------------------------------------------------------------
def smooth_noise_2d(rng, d1, d2, scale=0.15):
    """Low-freq 2D noise via Hanning smoothing (simple & controllable)."""

    base = rng.standard_normal((d1, d2)).astype(np.float32)
    wx = np.hanning(max(3, d1//6)).astype(np.float32)  # noqa
    wy = np.hanning(max(3, d2//6)).astype(np.float32)  # noqa
    wx = wx / (wx.sum() + 1e-8)
    wy = wy / (wy.sum() + 1e-8)
    from scipy.signal import convolve2d
    sm = convolve2d(base, np.outer(wx, wy), mode="same", boundary="wrap")

    return (scale * sm).astype(np.float32)  # noqa


# ----------------------------------------------------------------------------------------------------------------------
def _apply_xy_shift(grid_xy, shift_xy):

    sx, sy = shift_xy
    x, y = grid_xy
    x = np.clip(x + sx, -1, 1)
    y = np.clip(y + sy, -1, 1)

    return x, y


# ----------------------------------------------------------------------------------------------------------------------
def gen_image(shape: Shape3D, z, rng) -> np.ndarray:

    d1, d2, d3 = shape
    assert d3 == 1
    x = np.linspace(-1, 1, d1)[:, None]
    y = np.linspace(-1, 1, d2)[None, :]
    x, y = _apply_xy_shift((x, y), z["xy_shift"])
    r2 = x**2 + y**2
    theta = np.arctan2(y, x)
    blob = np.exp(-2.5 * r2)
    ridge = np.cos(3*theta + z["phase"]) * np.exp(-1.0 * r2)
    tex = smooth_noise_2d(rng, d1, d2, scale=z["texture"])
    img = z["contrast"] * (z["amp"] * blob + 0.5 * ridge) + tex + z["bias"]
    img = img.astype(np.float32)[..., None]

    return img


# ----------------------------------------------------------------------------------------------------------------------
def gen_video(shape: Shape3D, z, rng) -> np.ndarray:

    d1, d2, d3 = shape
    assert d3 > 1
    base = np.outer(np.hanning(d1), np.hanning(d2)).astype(np.float32)
    base += smooth_noise_2d(rng, d1, d2, scale=0.05)
    t = np.linspace(0, 2*np.pi, d3, endpoint=False)
    t = t + 2*np.pi*z["temporal_shift"]
    f = z["freq"]
    vid = []
    drift_x = rng.uniform(-0.01, 0.01)
    drift_y = rng.uniform(-0.01, 0.01)
    sx, sy = z["xy_shift"]

    for i, tt in enumerate(t):
        phase = z["phase"] + tt * f
        cosf = np.cos(phase) * base
        tex = smooth_noise_2d(rng, d1, d2, scale=z["texture"])
        xi = np.linspace(-1, 1, d1)[:, None] + (sx + i*drift_x)
        yi = np.linspace(-1, 1, d2)[None, :] + (sy + i*drift_y)
        xi = np.clip(xi, -1, 1)
        yi = np.clip(yi, -1, 1)
        r2 = xi**2 + yi**2
        ring = np.cos(4*np.sqrt(r2) + phase) * np.exp(-1.2*r2)
        frame = z["amp"] * (cosf + 0.3*ring) + tex + z["bias"]
        vid.append(frame.astype(np.float32))

    return np.stack(vid, axis=-1)


# ----------------------------------------------------------------------------------------------------------------------
def gen_timeseries(shape: Shape3D, z, rng) -> np.ndarray:

    _, _, d3 = shape
    t = np.linspace(0, 2*np.pi, d3, endpoint=False)
    f = z["freq"] * (1.0 + 0.1*rng.standard_normal())
    s = (z["amp"] * np.sin(f*t + z["phase"]) +
         0.3*np.sin(0.5*f*t + 0.7*z["phase"]) +
         0.05 * rng.standard_normal(d3) + z["bias"])
    out = np.zeros((1, 1, d3), dtype=np.float32)
    out[0, 0, :] = s.astype(np.float32)

    return out


# ----------------------------------------------------------------------------------------------------------------------
def gen_profile2d(shape: Shape3D, z, rng) -> np.ndarray:

    d1, _, d3 = shape
    t = np.linspace(0, 2*np.pi, d3, endpoint=False)
    out = np.zeros((d1, 1, d3), dtype=np.float32)
    base_f = z["freq"]
    for c in range(d1):
        f = base_f * (1.0 + 0.05*c)
        a = z["amp"] * (0.7 + 0.3*np.tanh(0.1*(c - d1/2)))
        phase = z["phase"] + 0.1*c
        out[c, 0, :] = (a*np.sin(f*t + phase) +
                        0.2*np.sin(0.5*f*t + 0.3*phase) +
                        0.03*rng.standard_normal(d3) + z["bias"]).astype(np.float32)

    return out


# ----------------------------------------------------------------------------------------------------------------------
def gen_vector(shape: Shape3D, z, rng) -> np.ndarray:

    d1, _, _ = shape
    x = np.linspace(0, 1, d1)
    v = (z["amp"] * np.sin(2*np.pi*z["freq"]*x + z["phase"]) +
         0.3*np.cos(4*np.pi*x + 0.5*z["phase"]) +
         0.05 * rng.standard_normal(d1) + z["bias"]).astype(np.float32)
    out = np.zeros((d1, 1, 1), dtype=np.float32)
    out[:, 0, 0] = v

    return out


# ----------------------------------------------------------------------------------------------------------------------
def gen_modal(shape: Shape3D, z, rng) -> np.ndarray:

    d1, d2, d3 = shape
    if d1 > 1 and d2 > 1 and d3 == 1:      # image
        return gen_image(shape, z, rng)
    if d1 > 1 and d2 > 1 and d3 > 1:       # video
        return gen_video(shape, z, rng)
    if d1 == 1 and d2 == 1 and d3 > 1:     # timeseries
        return gen_timeseries(shape, z, rng)
    if d1 > 1 and d2 == 1 and d3 > 1:      # profile2d
        return gen_profile2d(shape, z, rng)
    if d1 >= 1 and d2 == 1 and d3 == 1:    # vector
        return gen_vector(shape, z, rng)

    return rng.standard_normal(shape).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------------
def gen_pair(
    input_shapes: List[Shape3D],
    output_shapes: List[Shape3D],
    rng: np.random.Generator,
    nonlinear_coupling: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
):
    """
    Create correlated X and Y via a shared latent `z`, then (optionally) warp `z` for Y to avoid trivial identity
    mapping.
    """

    z = sample_latents(rng)
    z_y = nonlinear_coupling(z) if nonlinear_coupling is not None else z
    X = [gen_modal(shp, z, rng) for shp in input_shapes]
    Y = [gen_modal(shp, z_y, rng) for shp in output_shapes]

    return X, Y


# ----------------------------------------------------------------------------------------------------------------------
def mild_nonlinear_coupling(z: Dict[str, Any]) -> Dict[str, Any]:
    """A small, smooth warp of z to make Y differ from X while remaining correlated."""

    z = dict(z)
    z["amp"] *= 0.9 + 0.2*np.tanh(0.7*z["freq"])
    z["phase"] += 0.2*np.sin(3*z["freq"])
    z["freq"] *= 1.0 + 0.05*np.sin(z["phase"])
    z["xy_shift"] = np.array(z["xy_shift"]) * 0.8
    z["contrast"] *= 0.95
    z["temporal_shift"] *= 1.1

    return z


# ----------------------------------------------------------------------------------------------------------------------
def generate_raw_dataset(
    n_samples: int,
    input_shapes,
    target_shapes,
    seed: int | None = None,
    coupling_fn=None,
):
    """
    Auto-infer modality types from shapes and generate synthetic (X, Y) pairs coupled via a shared latent with optional
    nonlinear warp for Y.
    """

    rng = np.random.default_rng(seed)
    X_raw, Y_raw, latents = [], [], []

    for _ in range(n_samples):
        z = sample_latents(rng)
        z_y = coupling_fn(z) if coupling_fn else z

        X = [gen_modal(shp, z, rng) for shp in input_shapes]
        Y = [gen_modal(shp, z_y, rng) for shp in target_shapes]

        X_raw.append(X)
        Y_raw.append(Y)
        latents.append(z)

    return {"X_raw": X_raw, "Y_raw": Y_raw, "latents": latents}

# ----------------------------------------------------------------------------------------------------------------------
