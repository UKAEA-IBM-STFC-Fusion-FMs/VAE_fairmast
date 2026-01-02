"""
MultiModalFTTransformer
================================================================================
A flexible multimodal regression model with:

  • Pluggable input encoders (via an InputRegistry) that turn raw signals of various
    shapes/modalities into a single continuous feature vector.
  • A shared backbone (FT-Transformer if available, otherwise a small MLP).
  • Per-target prediction heads (via a TargetRegistry), each optionally paired with
    a decoder to reconstruct native output shapes from compact embeddings.

Typical flow
------------
1) Prepare InputRegistry with input specs (name, shape, modality, encoder_name, ...).
2) Prepare TargetRegistry with target specs (shape, encoder/decoder names, head size, ...).
3) Instantiate MultiModalFTTransformer(input_registry=..., target_registry=...).
4) Call model.forward(X_raw, active_targets, Y_for_meta=..., require_decoded=...).

Notes
-----
• Device/dtype: The model moves itself to `device`/`dtype` at init. Encoders run in fp32.
• FT-Transformer: uses `rtdl-revisiting-models` if installed; otherwise falls back to MLP.
• Decoding: If a target has a decoder, the head predicts an embedding whose size is given by
  decoder.get_embedding_dim(shape). Otherwise the head predicts the raw flattened tensor.
• Y_for_meta: If you need decoded outputs, pass a dict of reference tensors for the batch
  so encoders can produce metadata (e.g., B-spline basis, PCA components). These metadata
  are used by decoders at inference-time to invert the embedding.

Shape conventions
-----------------
• Inputs: user-defined. Each input spec must provide a shape (d1, d2, d3) so encoder dims
  can be computed (or flattened size if no encoder).
• Targets: each spec must provide `shape` (unless `out_dim_override` is set), so the head
  output size can be determined (decoder embedding size or raw flattened size).
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Callable
# from contextlib import nullcontext

import torch
import torch.nn as nn

# ======================================================================================================================
# Preliminaries

# Optional FT backbone
try:
    from rtdl_revisiting_models import FTTransformer
    _HAS_FT = True
except ImportError:
    FTTransformer = None
    print("\nPackage `rtdl_revisiting_models` not found: FT-Transformer block skipped!\n")
    _HAS_FT = False


# ----------------------------------------------------------------------------------------------------------------------
# Define supporting data types

Shape3D = Tuple[int, int, int]  # (d1, d2, d3)


# ----------------------------------------------------------------------------------------------------------------------
def _prod(s: Shape3D) -> int:
    return int(s[0] * s[1] * s[2])


# ======================================================================================================================
# Inputs
# ======================================================================================================================

# ======================================================================================================================
@dataclass
class InputSpec:
    """Specification for one input branch."""

    name: str
    shape: Optional[Shape3D] = None
    modality: Optional[str] = None
    encoder_name: Optional[str] = None
    encoder_kwargs: Optional[Dict[str, Any]] = None
    frozen: bool = False
    active: bool = True


# ======================================================================================================================
class InputRegistry:
    """
    Holds input specs and (optionally) encoder instances for each input.
    Encoders are created via the provided `get_encoder_fn` factory.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        specs: Dict[str, InputSpec],
        get_encoder: Callable[..., Any],
        infer_modality_from_shape_fn: Callable[[Shape3D], str],
    ):
        self.specs: Dict[str, InputSpec] = specs
        self.get_encoder = get_encoder
        self.infer_modality_from_shape = infer_modality_from_shape_fn
        self._names: List[str] = list(specs.keys())
        self.encoders: Dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------------------------------
    def names(self) -> List[str]:
        return list(self._names)

    # ------------------------------------------------------------------------------------------------------------------
    def active_names(self) -> List[str]:
        return [n for n in self._names if self.specs[n].active]

    # ------------------------------------------------------------------------------------------------------------------
    def bind_shapes(self, shapes_by_name: Dict[str, Shape3D]) -> None:
        """Attach concrete shapes to the declared inputs."""

        for n, shp in shapes_by_name.items():
            if n in self.specs:
                self.specs[n].shape = tuple(int(s) for s in shp)

    # ------------------------------------------------------------------------------------------------------------------
    def auto_fill_modalities(self) -> None:
        """Infer modality if missing, based on the bound shape."""

        for n, s in self.specs.items():
            if s.modality is None and s.shape is not None:
                s.modality = self.infer_modality_from_shape(s.shape)

    # ------------------------------------------------------------------------------------------------------------------
    def build_encoders(self, defaults_by_modality: Optional[Dict[str, Any]] = None) -> None:
        """Instantiate encoders from encoder_name/kwargs or modality defaults."""

        self.auto_fill_modalities()
        self.encoders.clear()
        for n, s in self.specs.items():
            if s.encoder_name is not None:
                self.encoders[n] = self.get_encoder(s.encoder_name, **(s.encoder_kwargs or {}))
            else:
                self.encoders[n] = (
                    defaults_by_modality.get(s.modality) if (defaults_by_modality and s.modality) else None
                )

    # ------------------------------------------------------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print summary info. """

        print("[inputs] name | shape | modality | encoder | active")
        for n in self._names:
            s = self.specs[n]
            enc = self.encoders.get(n, None)
            print("  ", (n, s.shape, s.modality, getattr(enc, "name", None), s.active))


# ======================================================================================================================
# Targets
# ======================================================================================================================

# ======================================================================================================================
@dataclass
class TargetSpec:
    """Specification for one target head/decoder branch."""

    name: str
    shape: Optional[Shape3D] = None
    encoder_name: Optional[str] = None     # Used only to create metadata for decoding
    encoder_kwargs: Optional[Dict[str, Any]] = None
    decoder_name: Optional[str] = None
    decoder_kwargs: Optional[Dict[str, Any]] = None
    head_hidden: int = 256
    out_dim_override: Optional[int] = None  # Force a specific head output size
    loss: str = "mse"
    loss_weight: float = 1.0


# ======================================================================================================================
class MLPHead(nn.Module):
    """Simple 2-layer MLP head for one target embedding/output."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ======================================================================================================================
class TargetRegistry(nn.Module):
    """
    Builds target encoders/decoders and heads from specs.

    • If decoder is provided, the head outputs decoder-embedding dim.
    • If no decoder, the head outputs the raw flattened target size.
    • If out_dim_override is set, it takes precedence.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        specs: Dict[str, TargetSpec],
        get_encoder: Callable[..., Any],
        get_decoder: Callable[..., Any],
    ):
        super().__init__()
        self.specs = specs
        self._get_encoder = get_encoder
        self._get_decoder = get_decoder

        self.decoders: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.heads = nn.ModuleDict()
        self.pred_dims: Dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------------------
    def names(self, only_with_shape: bool = False) -> List[str]:

        names = list(self.specs.keys())
        return [n for n in names if (self.specs[n].shape is not None)] if only_with_shape else names

    # ------------------------------------------------------------------------------------------------------------------
    def auto_fill_decoders(self) -> None:
        """If decoder not specified but an encoder is, mirror it by default."""

        for spec in self.specs.values():
            if spec.decoder_name is None and spec.encoder_name is not None:
                spec.decoder_name = spec.encoder_name
                spec.decoder_kwargs = dict(spec.encoder_kwargs or {})

    # ------------------------------------------------------------------------------------------------------------------
    def bind_shapes(self, shapes_by_name: Dict[str, Shape3D]) -> None:
        """Attach concrete shapes to targets."""

        for name, shp in shapes_by_name.items():
            if name not in self.specs:
                raise KeyError(f"Unknown target '{name}' in shapes_by_name.")
            self.specs[name].shape = tuple(map(int, shp))

    # ------------------------------------------------------------------------------------------------------------------
    def build(self, trunk_out_dim: int) -> nn.ModuleDict:
        """Instantiate encoders/decoders/heads and compute per-target output dims."""

        self.auto_fill_decoders()
        for name, spec in self.specs.items():
            if spec.shape is None and spec.out_dim_override is None:
                raise ValueError(
                    f"Target '{name}' missing shape/out_dim_override. Call bind_shapes() first."
                )

            # Decoder (optional)
            dec = self._get_decoder(spec.decoder_name, **(spec.decoder_kwargs or {})) \
                if spec.decoder_name is not None else None
            self.decoders[name] = dec

            # Encoder for metadata (optional)
            enc = self._get_encoder(spec.encoder_name, **(spec.encoder_kwargs or {})) \
                if spec.encoder_name is not None else None
            self.encoders[name] = enc

            # Output size
            if spec.out_dim_override is not None:
                out_dim = int(spec.out_dim_override)
            elif dec is not None:
                if not hasattr(dec, "get_embedding_dim"):
                    raise ValueError(f"Decoder for '{name}' lacks get_embedding_dim; set out_dim_override.")
                out_dim = int(dec.get_embedding_dim(spec.shape))
            else:
                out_dim = _prod(spec.shape)

            self.pred_dims[name] = out_dim
            self.heads[name] = MLPHead(trunk_out_dim, out_dim, hidden=spec.head_hidden)

        return self.heads


# ======================================================================================================================
# Model
# ======================================================================================================================

# ======================================================================================================================
class MultiModalFTTransformer(nn.Module):
    """
    Multimodal regressor with optional FT-Transformer backbone and per-target heads/decoders.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        input_registry: InputRegistry,  # Prepared with bound shapes and encoders (or defaults)
        device: Optional[torch.device] = None,
        n_blocks: int = 3,  # FT-Transformer depth if available
        target_registry: Optional[TargetRegistry] = None,
        verbose: bool = False,
    ):
        super().__init__()
        if target_registry is None:
            raise ValueError("target_registry must be provided.")
        self.verbose = verbose
        self.device = device or torch.device("cpu")

        # ..............................................................................................................
        # Inputs (from InputRegistry)
        # ..............................................................................................................

        self.input_registry = input_registry
        if getattr(self.input_registry, "encoders", None) in (None, {}):
            self.input_registry.build_encoders()

        self.input_names: List[str] = self.input_registry.names()
        self.input_shapes: List[Shape3D] = [self.input_registry.specs[n].shape for n in self.input_names]
        self.input_modalities: List[Optional[str]] = [self.input_registry.specs[n].modality for n in self.input_names]
        self.input_encoders_by_name: Dict[str, Any] = dict(self.input_registry.encoders)

        for n, shp in zip(self.input_names, self.input_shapes):
            if shp is None:
                raise ValueError(f"Input '{n}' is missing a bound shape.")

        self.input_split_sizes: List[int] = []
        for n, shp in zip(self.input_names, self.input_shapes):
            enc = self.input_encoders_by_name.get(n, None)
            if enc is None:
                d1, d2, d3 = shp
                self.input_split_sizes.append(int(d1 * d2 * d3))
            else:
                self.input_split_sizes.append(int(enc.get_embedding_dim(shp)))
        self.total_input_dim = int(sum(self.input_split_sizes))

        # ..............................................................................................................
        # Backbone
        # ..............................................................................................................

        if _HAS_FT:
            bb_kwargs = FTTransformer.get_default_kwargs(n_blocks=n_blocks)
            bb_kwargs["d_out"] = None  # We will attach our own heads
            self.backbone = FTTransformer(
                n_cont_features=self.total_input_dim,
                cat_cardinalities=[],
                **bb_kwargs
            )
            try:
                self.trunk_out_dim = self.backbone.backbone.blocks[0]["ffn"].linear2.out_features
            except Exception as ee:
                if verbose:
                    print(f"Unexpected exception while calculating `self.trunk_out_dim`: {ee}")
                self.trunk_out_dim = max(128, self.total_input_dim)
            self._has_ft = True
        else:
            self.trunk_out_dim = max(128, self.total_input_dim)
            self.backbone = nn.Sequential(
                nn.Linear(self.total_input_dim, self.trunk_out_dim),
                nn.ReLU(),
                nn.Linear(self.trunk_out_dim, self.trunk_out_dim),
                nn.ReLU(),
            )
            self._has_ft = False

        # ..............................................................................................................
        # Targets (registry)
        # ..............................................................................................................

        self.registry: TargetRegistry = target_registry
        self.heads = self.registry.build(trunk_out_dim=int(self.trunk_out_dim))
        self.decoders = self.registry.decoders
        self.encoders = self.registry.encoders
        self.pred_dims = dict(self.registry.pred_dims)  # name -> out_dim
        self.target_shapes = {n: spec.shape for n, spec in self.registry.specs.items()}
        self.total_pred_dim = int(sum(self.pred_dims.values()))

        # Move params to device ONLY; keep fp32 dtype
        self.to(self.device)

        if self.verbose:
            self._print_init_summary()

    # ------------------------------------------------------------------------------------------------------------------
    # Summary
    def _print_init_summary(self) -> None:
        """Print init summary."""

        print(f"\n[init] device={self.device} (params are fp32)")

        print("[init] inputs : (name, modality, shape, embed_dim, encoder)")
        for n, shp, mod, d in zip(self.input_names, self.input_shapes, self.input_modalities, self.input_split_sizes):
            enc = self.input_encoders_by_name.get(n, None)
            enc_name = getattr(enc, "name", None)
            print("  ", (n, mod, shp, d, enc_name))

        pred_specs = []
        for name in sorted(self.registry.specs.keys()):
            spec = self.registry.specs[name]
            shp = spec.shape
            pred_dim = self.pred_dims[name]
            raw_sz = _prod(shp) if shp is not None else None
            kind = "embed" if (self.decoders.get(name) is not None) else "raw"
            pred_specs.append((name, shp, pred_dim, f"{kind}" + (f" (raw={raw_sz})" if raw_sz is not None else "")))
        print(f"[init] targets: {pred_specs}")

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[init] params={n_params/1e6:.2f}M | total_in={self.total_input_dim} "
              f"→ sum_head_out={self.total_pred_dim}")

    # ------------------------------------------------------------------------------------------------------------------
    # Utilities
    def _to_torch(self, x: Any, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
        """Move/convert `x` to model device; default dtype is fp32."""

        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------------------------------------------------------
    # Encode inputs
    def encode_inputs(self, X_raw: List[List[Any]]) -> torch.Tensor:
        """
        Convert raw per-input signals into a single (B, total_input_dim) fp32 tensor.

        X_raw: list over batch of lists over inputs.
            The inner order MUST match InputRegistry.names().
        """

        B = len(X_raw)
        if B == 0:
            return torch.empty((0, self.total_input_dim), device=self.device, dtype=torch.float32)

        L = len(self.input_names)
        for b, row in enumerate(X_raw):
            if len(row) != L:
                raise ValueError(f"Sample #{b} has {len(row)} inputs, expected {L}.")

        chunks: List[torch.Tensor] = []
        for i, name in enumerate(self.input_names):
            enc = self.input_encoders_by_name.get(name, None)
            embed_b: List[torch.Tensor] = []

            for b in range(B):
                sig = X_raw[b][i]
                sig_t = self._to_torch(sig, dtype=torch.float32)  # Keep fp32 for encoders
                if enc is None:
                    emb = sig_t.flatten()
                else:
                    meta = enc.encode(sig_t)  # sig_t is already fp32
                    emb = meta["embedding"].reshape(-1)
                embed_b.append(emb)

            chunks.append(torch.stack(embed_b, dim=0))  # (B, dim_i)

        return torch.cat(chunks, dim=1)  # (B, total_input_dim) fp32

    # ------------------------------------------------------------------------------------------------------------------
    # Encode output metadata
    def encode_output_metadata(
        self,
        Y_for_meta: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Build per-sample metadata dicts for each target using target encoders.

        Y_for_meta:
          dict[target_name] -> tensor with shape (B, d1, d2, d3)
          Only needed when require_decoded=True (for decoders that need metadata).
        """

        if not Y_for_meta:
            return None

        names = list(Y_for_meta.keys())
        B = Y_for_meta[names[0]].shape[0]
        out: List[Dict[str, Any]] = [dict() for _ in range(B)]

        for tname, Yt in Y_for_meta.items():
            enc = self.encoders.get(tname)
            if enc is None:
                for b in range(B):
                    out[b][tname] = {"shape": tuple(Yt[b].shape)}
                continue

            for b in range(B):
                y_b = self._to_torch(Yt[b], dtype=torch.float32)
                meta = enc.encode(y_b)
                if isinstance(meta, dict) and "embedding" in meta:
                    meta = {k: v for k, v in meta.items() if k != "embedding"}
                out[b][tname] = meta

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Decode per target
    def decode_per_target(
        self,
        preds_per_target: Dict[str, torch.Tensor],
        meta_batch: Optional[List[Dict[str, Any]]],
        specs: Dict[str, TargetSpec],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode predictions back to native shapes per target.
        If no decoder is present, reshape flat predictions to (B, d1, d2, d3).
        """

        decoded: Dict[str, torch.Tensor] = {}
        for tname, pred in preds_per_target.items():
            B = pred.shape[0]
            spec = specs[tname]
            dec = self.decoders.get(tname)

            if dec is None:
                d1, d2, d3 = spec.shape
                decoded[tname] = pred.view(B, d1, d2, d3)
            else:
                outs = []
                for b in range(B):
                    emb_b = pred[b].reshape(-1)
                    meta_b = None if meta_batch is None else meta_batch[b][tname]
                    y_b = dec.decode_torch(emb_b, meta_b)  # decoder runs fp32
                    outs.append(self._to_torch(y_b, dtype=torch.float32))
                decoded[tname] = torch.stack(outs, dim=0)

        return decoded

    # ------------------------------------------------------------------------------------------------------------------
    # Backbone forward
    def backbone_forward(self, X_raw: List[List[Any]]) -> torch.Tensor:
        """Encode inputs and run the shared backbone (fp32)."""

        X = self.encode_inputs(X_raw)
        if _HAS_FT:
            return self.backbone(x_cat=None, x_cont=X)
        return self.backbone(X)

    # ------------------------------------------------------------------------------------------------------------------
    # Backbone forward no-grad
    @torch.no_grad()
    def backbone_forward_nograd(self, X_raw: List[List[Any]]) -> torch.Tensor:
        """Backbone forward with no grad (e.g., evaluation embeddings)."""

        return self.backbone_forward(X_raw)

    # ------------------------------------------------------------------------------------------------------------------
    # Full forward
    def forward(
        self,
        X_raw: List[List[Any]],
        active_targets: List[str],
        Y_for_meta: Optional[Dict[str, torch.Tensor]] = None,
        require_decoded: bool = False,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        X_raw : list[list[Any]]
            Batch of per-input raw signals. Order must match InputRegistry.names().
        active_targets : list[str]
            Subset of targets to predict this step. (e.g., for curriculum/masking)
        Y_for_meta : dict[str, Tensor] or None
            Optional ground-truth tensors (B, d1, d2, d3) used only to compute decoding metadata. Required if
            `require_decoded` and your decoders need per-batch metadata (e.g., B-splines, PCA).
        require_decoded : bool
            If True, also return native-shape predictions under out["decoded"].

        Returns
        -------
        dict with keys:
          - "z": shared backbone features (B, trunk_out_dim)
          - "preds": dict[target_name] = flat predictions (B, pred_dim)
          - "decoded": (optional) dict[target_name] = native-shape predictions (B, d1, d2, d3)
        """
        
        # Validate active targets up front
        unknown = [t for t in active_targets if t not in self.registry.specs]
        if unknown:
            raise KeyError(f"Unknown active_targets: {unknown}")

        z = self.backbone_forward(X_raw)
        preds: Dict[str, torch.Tensor] = {t: self.heads[t](z) for t in active_targets}  # fp32
        out: Dict[str, Any] = {"z": z, "preds": preds}

        if require_decoded:
            meta_batch = self.encode_output_metadata(Y_for_meta)
            decoded = self.decode_per_target(preds, meta_batch, self.registry.specs)
            out["decoded"] = decoded

        return out

    # ------------------------------------------------------------------------------------------------------------------
