"""
Utilities to encode raw signals (time series, 2D profiles, images, videos) into compact, model-friendly embeddings, and
to decode model outputs back to the native signal space.

Design goals:
  - Torch-first: encoders return metadata and, when relevant, a torch.Tensor "embedding".
  - Decoders support decode_torch() for autograd-safe reconstruction and a legacy decode() convenience that returns
    numpy (handy for evaluation/visualization, not for training).
  - Registry pattern: plug-and-play new encoders/decoders with @register_* decorators.
  - Shape conventions:
      * 1D flattened signals: (S,) treated as (1, 1, S) logically
      * 2D profiles per-channel: (C, 1, T)
      * Images: (H, W) or (H, W, 1)
      * Videos: (H, W, T)

What “DCT” stands for:
  - DCT = Discrete Cosine Transform (we use orthonormal DCT-II).

Available Encoders (get_encoder("<name>", **kwargs)):
  - "flatten_bspline_1d"        -> FlattenBsplineEncoder1D
  - "per_channel_bspline_1d"    -> PerChannelBsplineEncoder1D (for (C,1,T))
  - "fpca_3d"                   -> FPCA3DEncoder (also used by FPCA2DEncoder)
  - "fpca_2d"                   -> FPCA2DEncoder (2D specialization)
  - "dct_2d"                    -> DCT2DEncoder  (images)
  - "dct_3d"                    -> DCT3DEncoder  (videos)

Available Decoders (get_decoder("<name>", **kwargs)):
  - "identity"                  -> IdentityDecoder
  - "linear_map"                -> LinearMapDecoder (generic y = b + W^T coeffs)
  - "flatten_bspline_1d"        -> FlattenBsplineDecoder1D
  - "per_channel_bspline_1d"    -> PerChannelBsplineDecoder1D
  - "fpca_3d"                   -> FPCA3DDecoder (also used by FPCA2DDecoder)
  - "fpca_2d"                   -> FPCA2DDecoder
  - "dct_2d"                    -> DCT2DDecoder
  - "dct_3d"                    -> DCT3DDecoder

Notes on device support:
  - Some linear algebra (QR in torch.pca_lowrank, torch.linalg.lstsq) is not supported on Apple  MPS. We automatically
  and safely fall back to CPU for those bits, then move results back to the original device/dtype. This is encapsulated
  in _pca_topk_lowrank() and _lstsq_cpu_safe().

Synthetic tests:
  - At the bottom of the file there is a commented-out section that generates synthetic data, runs encoders/decoders,
    and visualizes reconstructions. Keep it around: it’s very convenient to sanity-check new encoders/decoders quickly.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import torch
import math
import torch.nn.functional as F  # noqa
import contextlib

try:
    _autocast = torch.autocast  # PyTorch >= 1.12
except AttributeError:
    try:
        from torch.cuda.amp import autocast as _autocast  # Older fallback
    except ImportError:
        _autocast = None


# Required imports for testing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from scipy.interpolate import splrep, BSpline, splev


# ----------------------------------------------------------------------------------------------------------------------
def generate_single_channel_video(height, width, num_frames):

    video_ = np.zeros((height, width, num_frames))
    for h in range(height):
        for w in range(width):
            phase = h * 0.1 + w * 0.2
            video_[h, w, :] = (
                    0.6 * np.sin(2 * np.pi * grid / num_frames + phase) +
                    0.3 * np.sin(4 * np.pi * grid / num_frames + 0.5 * phase)
            )

    return video_


# ----------------------------------------------------------------------------------------------------------------------
def _amp_off():

    # Only matters on CUDA; on CPU/MPS this becomes a no-op
    if _autocast is None:
        return contextlib.nullcontext()
    return _autocast(device_type="cuda", enabled=False)


# ======================================================================================================================
# Registry
# ======================================================================================================================

ENCODER_REGISTRY: Dict[str, type] = {}
DECODER_REGISTRY: Dict[str, type] = {}


# ----------------------------------------------------------------------------------------------------------------------
def register_encoder(name: str):
    """Decorator to register an encoder class by name."""

    def _wrap(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return _wrap


# ----------------------------------------------------------------------------------------------------------------------
def register_decoder(name: str):
    """Decorator to register a decoder class by name."""

    def _wrap(cls):
        DECODER_REGISTRY[name] = cls
        return cls
    return _wrap


# ----------------------------------------------------------------------------------------------------------------------
def get_encoder(name: Optional[str], **kwargs):
    """Factory: build an encoder instance from the registry."""

    if name is None:
        return None
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Encoder '{name}' not found in registry.")
    return ENCODER_REGISTRY[name](**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
def get_decoder(name: Optional[str], **kwargs):
    """Factory: build a decoder instance from the registry."""

    if name is None:
        return None
    if name not in DECODER_REGISTRY:
        raise ValueError(f"Decoder '{name}' not found in registry.")
    return DECODER_REGISTRY[name](**kwargs)

# ======================================================================================================================
# Base classes
# ======================================================================================================================


# ======================================================================================================================
class BaseEncoder:

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:
        """
        Convert a raw signal (torch.Tensor or numpy.ndarray) to a metadata dict.
        If 'embedding' is present, it MUST be a torch.Tensor (training-friendly).
        """

        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """Return the embedding dimension for a given native input shape."""

        raise NotImplementedError


# ======================================================================================================================
class BaseDecoder:

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """Return the expected embedding dimension for a given native output shape."""

        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Torch-first API (keeps autograd for differentiable decode)
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Legacy convenience (NOT for training / autograd). Returns numpy.
    def decode(self, embedding, metadata: Dict[str, Any]):

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
        out = self.decode_torch(embedding, metadata)
        return out.detach().cpu().numpy()


# ======================================================================================================================
# Utils
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def _as_torch(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """Coerce input to torch.Tensor on the desired device/dtype."""

    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


# ----------------------------------------------------------------------------------------------------------------------
# B-spline basis (Cached)

_BSPLINE_CACHE: Dict[Tuple[int, int, int, str, str], torch.Tensor] = {}


# ----------------------------------------------------------------------------------------------------------------------
def _uniform_open_knot_vector(domain_start: float, domain_end: float, degree: int, num_basis: int, device, dtype):
    """
    Build an open (clamped) uniform knot vector suitable for B-spline basis construction.
    Ensures num_basis >= degree + 1.
    """

    if num_basis < degree + 1:
        raise ValueError(f"num_basis ({num_basis}) must be >= degree+1 ({degree+1}).")
    n_internal = num_basis - degree - 1
    start = torch.full((degree + 1,), float(domain_start), device=device, dtype=dtype)
    end = torch.full((degree + 1,), float(domain_end),   device=device, dtype=dtype)
    if n_internal > 0:
        # Strictly inside the domain
        internal = torch.linspace(domain_start + 1.0, domain_end - 1.0, n_internal, device=device, dtype=dtype)
        return torch.cat([start, internal, end], dim=0)

    return torch.cat([start, end], dim=0)


# ----------------------------------------------------------------------------------------------------------------------
def _compute_bspline_basis_matrix_1d(S: int, degree: int, num_basis: int, device, dtype) -> torch.Tensor:
    """
    Compute the 1D B-spline basis matrix Phi of shape (S, K), K=num_basis, over a grid x=0, ..., S-1.
    Uses Cox–de Boor recursion. Returns contiguous (S,K) on the requested device/dtype.
    """

    if S < 1:
        raise ValueError("S must be >= 1")
    x = torch.arange(S, device=device, dtype=dtype)  # (S,)
    knots = _uniform_open_knot_vector(0.0, float(S - 1), degree, num_basis, device, dtype)
    K = num_basis

    # Degree 0 indicator basis
    N = torch.zeros((K, S), device=device, dtype=dtype)
    for i in range(K):
        if i == K - 1:
            mask = (x >= knots[i]) & (x <= knots[i + 1])
        else:
            mask = (x >= knots[i]) & (x < knots[i + 1])
        N[i, mask] = 1.0

    # Cox–de Boor recursion up to the desired degree
    for p in range(1, degree + 1):
        N_next = torch.zeros_like(N)
        for i in range(K):
            # left term
            denom_left = knots[i + p] - knots[i]
            left = ((x - knots[i]) / denom_left) * N[i, :] if denom_left > 0 else 0.0
            # right term
            if i + 1 < K:
                denom_right = knots[i + p + 1] - knots[i + 1]
                right = ((knots[i + p + 1] - x) / denom_right) * N[i + 1, :] if denom_right > 0 else 0.0
            else:
                right = 0.0
            N_next[i, :] = left + right
        N = N_next

    return N.t().contiguous()  # (S, K)


# ----------------------------------------------------------------------------------------------------------------------
def bspline_basis_matrix_1d(S: int, degree: int, num_basis: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Cache-backed wrapper around _compute_bspline_basis_matrix_1d."""

    key = (int(S), int(degree), int(num_basis), str(device), str(dtype))
    Phi = _BSPLINE_CACHE.get(key)
    if Phi is None:
        Phi = _compute_bspline_basis_matrix_1d(S, degree, num_basis, device, dtype)
        _BSPLINE_CACHE[key] = Phi
    return Phi


# ======================================================================================================================
# Helpers
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
def _pca_topk_lowrank(Xc: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Low-rank PCA on centered data Xc (samples, features).
    Returns:
        scores:    (samples, k)
        components:(k, features)
    Notes:
      - Picks a safe oversampling q for torch.pca_lowrank.
      - Handles MPS by running pca_lowrank on CPU then moving results back.
    """

    with _amp_off():

        # Choose q >= k safely
        m, n = Xc.shape
        q_oversample = max(k + 2, int(1.25 * k))
        q = min(m, n, q_oversample)
        q = max(k, q)

        # MPS fallback for QR inside pca_lowrank
        orig_device, orig_dtype = Xc.device, Xc.dtype
        if orig_device.type == "mps":
            Xc_cpu = Xc.detach().to("cpu", dtype=torch.float32)
            U, S, V = torch.pca_lowrank(Xc_cpu, q=q, center=False)   # V: (features, q) on CPU
            comps = V[:, :k].T.contiguous().to(orig_device, dtype=orig_dtype)  # (k, features)
        else:
            U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
            comps = V[:, :k].T.contiguous()  # (k, features)

        # Scores on original device
        scores = Xc @ comps.T  # (samples, k)
        return scores, comps


# ----------------------------------------------------------------------------------------------------------------------
def _lstsq_cpu_safe(Phi: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Solve least-squares min ||Phi @ X - Y|| using torch.linalg.lstsq.

    Works on:
      - GPU/CPU natively
      - Apple MPS by temporarily moving to CPU (then returning X to the original device/dtype)

    Supports:
      - Y shape (N,)  -> X shape (K,)
      - Y shape (N,R) -> X shape (K,R)
    """
    with _amp_off():
        orig_device, orig_dtype = Phi.device, Phi.dtype

        if orig_device.type == "mps":
            Phi_cpu = Phi.detach().to("cpu", dtype=torch.float32)
            Y_cpu = Y.detach().to("cpu", dtype=torch.float32)
            X_cpu = torch.linalg.lstsq(Phi_cpu, Y_cpu).solution
            return X_cpu.to(orig_device, dtype=orig_dtype)
        else:
            return torch.linalg.lstsq(Phi, Y).solution


# ======================================================================================================================
# Identity (predict original directly)
# ======================================================================================================================

# ======================================================================================================================
@register_decoder("identity")
class IdentityDecoder(BaseDecoder):
    """No-op decoder that reshapes a flat embedding back to the original signal shape."""

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        d1, d2, d3 = input_shape
        return d1 * d2 * d3

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        shape = tuple(int(s) for s in metadata["shape"])
        return embedding.view(*shape)


# ======================================================================================================================
# Generic linear map (FPCA-like)
# ======================================================================================================================

# ======================================================================================================================
@register_decoder("linear_map")
class LinearMapDecoder(BaseDecoder):
    """
    Linear decoder: y = b + W^T @ coeffs  (implemented via F.linear).
    Expects metadata to contain:
      - "shape": (d1,d2,d3)
      - "W": (K, N)  torch-like array (will be coerced to torch)
      - "b": (N,)    torch-like array
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.name = "linear_map"

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        # Requires metadata to know K; not used for global dimension inference.
        raise NotImplementedError("linear_map.get_embedding_dim needs metadata; avoid calling globally.")

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        W = _as_torch(metadata["W"], device=embedding.device, dtype=embedding.dtype)  # (K, N)
        b = _as_torch(metadata["b"], device=embedding.device, dtype=embedding.dtype)  # (N,)
        K = W.shape[0]
        emb2 = embedding.view(-1, K)  # (samples, K) or (1,K)
        # F.linear: out = emb2 @ W.T + b  -> (samples, N)
        y_flat = F.linear(emb2, W, b)  # (samples, N)
        if y_flat.shape[0] == 1:
            y_flat = y_flat.view(-1)
        else:
            y_flat = y_flat.reshape(-1)
        return y_flat.view(*metadata["shape"])


# ======================================================================================================================
# Flatten B-spline 1D
# ======================================================================================================================

# ======================================================================================================================
@register_encoder("flatten_bspline_1d")
class FlattenBsplineEncoder1D(BaseEncoder):
    """
    Fit a global 1D B-spline basis to a flattened signal y (S,) and return coefficients (K,).
    Metadata includes Phi for exact reconstruction.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_basis: int = 5, degree: int = 3):
        self.name = "flatten_bspline_1d"
        self.num_basis = int(num_basis)
        self.degree = int(degree)

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:
        y = _as_torch(signal, dtype=torch.float32)
        shape = tuple(y.shape)
        y = y.flatten()  # (S,)
        S = y.numel()
        device, dtype = y.device, y.dtype

        Phi = bspline_basis_matrix_1d(S, self.degree, self.num_basis, device=device, dtype=dtype)  # (S,K)
        coeffs = _lstsq_cpu_safe(Phi, y)  # (K,)

        return {
            "embedding": coeffs,            # (K,)
            "shape": shape,                 # (d1,d2,d3)
            "Phi": Phi,                     # (S,K)
            "degree": self.degree,
            "num_basis": self.num_basis,
            "encoder_name": self.name,
            "method": "bspline_flatten",
        }

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        return self.num_basis


# ======================================================================================================================
@register_decoder("flatten_bspline_1d")
class FlattenBsplineDecoder1D(BaseDecoder):
    """Reconstruct flattened signals using stored Phi @ coeffs."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_basis: int = 5, degree: int = 3):
        self.name = "flatten_bspline_1d"
        self.num_basis = int(num_basis)
        self.degree = int(degree)

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_encoder(cls, encoder: FlattenBsplineEncoder1D):
        return cls(num_basis=encoder.num_basis, degree=encoder.degree)

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        return self.num_basis

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        Phi = _as_torch(metadata["Phi"], device=embedding.device, dtype=embedding.dtype)  # (S,K)
        shape = tuple(int(s) for s in metadata["shape"])
        y_flat = Phi @ embedding.view(-1)  # (S,)
        return y_flat.view(*shape)


# ======================================================================================================================
# Per-channel B-spline 1D  (for (C,1,T))
# ======================================================================================================================

# ======================================================================================================================
@register_encoder("per_channel_bspline_1d")
class PerChannelBsplineEncoder1D(BaseEncoder):
    """
    Fit the SAME 1D B-spline basis to each channel of a (C,1,T) profile.
    Returns concatenated coefficients of shape (C*K,). Metadata stores Phi and C.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, degree: int = 3, num_basis: int = 5):
        self.name = "per_channel_bspline_1d"
        self.degree = int(degree)
        self.num_basis = int(num_basis)
        self.num_channels: Optional[int] = None

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:
        """
        signal: (C, 1, T)
        Returns flattened coeffs of shape (C*K,)
        """

        y = _as_torch(signal, dtype=torch.float32)
        assert y.ndim == 3 and y.shape[1] == 1, "Expected shape (C,1,T)"
        C, _, T = y.shape
        self.num_channels = C
        device, dtype = y.device, y.dtype

        Phi = bspline_basis_matrix_1d(T, self.degree, self.num_basis, device=device, dtype=dtype)  # (T,K)
        # Solve Phi @ C ≈ Y with Y: (T,C) (vectorized least squares for all channels)
        Y = y[:, 0, :].t().contiguous()               # (T, C)
        sol = _lstsq_cpu_safe(Phi, Y)                 # (K, C)
        coeffs = sol.t().reshape(-1)                  # (C*K,)

        return {
            "embedding": coeffs,            # (C*K,)
            "shape": (C, 1, T),
            "Phi": Phi,                     # (T,K)
            "degree": self.degree,
            "num_basis": self.num_basis,
            "num_channels": C,
            "encoder_name": self.name,
            "method": "bspline_per_channel",
        }

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        C, _, _ = input_shape
        return C * self.num_basis


# ======================================================================================================================
@register_decoder("per_channel_bspline_1d")
class PerChannelBsplineDecoder1D(BaseDecoder):
    """Reconstruct (C,1,T) profiles from concatenated per-channel B-spline coefficients."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_basis: int = 5, degree: int = 3):
        self.name = "per_channel_bspline_1d"
        self.num_basis = int(num_basis)
        self.degree = int(degree)

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_encoder(cls, encoder: PerChannelBsplineEncoder1D):
        return cls(num_basis=encoder.num_basis, degree=encoder.degree)

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        C, _, _ = input_shape
        return C * self.num_basis

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        C, _, T = metadata["shape"]
        Phi = _as_torch(metadata["Phi"], device=embedding.device, dtype=embedding.dtype)  # (T,K)
        K = Phi.shape[1]
        emb = embedding.view(C, K)                    # (C, K)
        # Vectorized reconstruction: (C,K) @ (K,T) -> (C,T)
        recon = emb @ Phi.t()                         # (C, T)
        out = torch.zeros((C, 1, T), device=embedding.device, dtype=embedding.dtype)
        out[:, 0, :] = recon
        return out


# ======================================================================================================================
# FPCA 3D / 2D (torch PCA low-rank)
# ======================================================================================================================

# ======================================================================================================================
@register_encoder("fpca_3d")
class FPCA3DEncoder(BaseEncoder):
    """
    FPCA-like encoding via low-rank PCA over either 'space' or 'time' axes.
    Input: (H, W, T)
    pca_dim:
      - 'time'  -> PCA over (H*W, T)   (samples=H*W, features=T)
      - 'space' -> PCA over (T, H*W)   (samples=T,   features=H*W)
    compression: optional spatial stride compression (h_comp,w_comp). Decode reconstructs the
                 compressed grid; any upsampling must be handled externally.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self, num_components: int = 5,
            compression: Optional[Tuple[int, int]] = None,
            pca_dim: str = "space"
    ):
        self.name = "fpca_3d"
        self.num_components = int(num_components)
        self.compression = compression
        self.pca_dim = pca_dim

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:
        Y = _as_torch(signal, dtype=torch.float32)
        H, W, T = Y.shape

        # Optional coarse spatial compression by striding.
        if self.compression is not None:
            h_comp, w_comp = self.compression
            h_stride = max(1, H // int(h_comp))
            w_stride = max(1, W // int(w_comp))
            Y = Y[::h_stride, ::w_stride, :]
            H, W = Y.shape[:2]

        if self.pca_dim == "time":
            X = Y.reshape(-1, T)            # (H*W, T)
        elif self.pca_dim == "space":
            X = Y.reshape(-1, T).t()        # (T, H*W)
        else:
            raise ValueError(f"Unknown pca_dim: {self.pca_dim}")

        mean = X.mean(dim=0, keepdim=True)  # (1, features)
        Xc = X - mean

        k = min(self.num_components, min(Xc.shape))
        scores, comps = _pca_topk_lowrank(Xc, k)  # scores: (samples,k), comps: (k, features)

        # Flatten scores for a simple embedding (handy when routing through a tabular model).
        embedding = scores.reshape(-1)  # (samples*k,)

        return {
            "embedding": embedding,         # flattened (samples*k,)
            "shape": (H, W, T),
            "pca_components": comps,        # (k, features)
            "pca_mean": mean.squeeze(0),    # (features,)
            "n_components": k,
            "pca_dim": self.pca_dim,
            "encoder_name": self.name,
        }

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, T = input_shape
        if self.pca_dim == "space":
            return self.num_components * T
        elif self.pca_dim == "time":
            return self.num_components * (H * W)
        else:
            raise ValueError(f"Unknown pca_dim: {self.pca_dim}")


# ======================================================================================================================
@register_decoder("fpca_3d")
class FPCA3DDecoder(BaseDecoder):
    """Inverse of FPCA3DEncoder using stored components and mean."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_components: int = 5, pca_dim: str = "space"):
        self.name = "fpca_3d"
        self.num_components = int(num_components)
        self.pca_dim = pca_dim

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_encoder(cls, encoder: FPCA3DEncoder):
        return cls(num_components=encoder.num_components, pca_dim=encoder.pca_dim)

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, T = input_shape
        if self.pca_dim == "space":
            return self.num_components * T
        elif self.pca_dim == "time":
            return self.num_components * (H * W)
        else:
            raise ValueError(f"Unknown pca_dim: {self.pca_dim}")

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:

        H, W, T = metadata["shape"]
        comps = _as_torch(metadata["pca_components"], device=embedding.device, dtype=embedding.dtype)  # (k, features)
        mean = _as_torch(metadata["pca_mean"], device=embedding.device, dtype=embedding.dtype)  # (features,)
        k, dim = comps.shape

        # Reshape to (samples,k) according to pca_dim
        if metadata["pca_dim"] == "space":
            samples = T
        elif metadata["pca_dim"] == "time":
            samples = H * W
        else:
            raise ValueError(f"Unknown pca_dim: {metadata['pca_dim']}")

        emb2 = embedding.view(samples, k)  # (samples, k)

        # X_rec = emb2 @ comps + mean  (F.linear handles the bias addition)
        X_rec = F.linear(emb2, comps.t(), mean)  # (samples, dim)

        if metadata["pca_dim"] == "time":   # (H*W, T) → (H, W, T)
            return X_rec.view(H, W, T)
        else:                                # (T, H*W) → (T, H, W) → (H, W, T)
            return X_rec.view(T, H, W).permute(1, 2, 0).contiguous()


# ======================================================================================================================
@register_encoder("fpca_2d")
class FPCA2DEncoder(FPCA3DEncoder):
    """2D specialization of FPCA3DEncoder (no compression) on (H,W,1) or (H,W) treated as (H,W,T)."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_components: int = 5, pca_dim: str = "space"):
        super().__init__(num_components=num_components, compression=None, pca_dim=pca_dim)


# ======================================================================================================================
@register_decoder("fpca_2d")
class FPCA2DDecoder(FPCA3DDecoder):
    """2D specialization of FPCA3DDecoder."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, num_components: int = 5, pca_dim: str = "space"):
        super().__init__(num_components=num_components, pca_dim=pca_dim)


# ======================================================================================================================
# DCT utilities (orthonormal DCT-II) + Encoders/Decoders
# ======================================================================================================================

# Cache for 1D DCT matrices
_DCT_CACHE: Dict[Tuple[int, str, str], torch.Tensor] = {}


# ----------------------------------------------------------------------------------------------------------------------
def dct_matrix_1d(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix B of shape (N, N) such that:
        forward:  c = B @ x
        inverse:  x = B.T @ c
    """
    key = (int(N), str(device), str(dtype))
    B = _DCT_CACHE.get(key)
    if B is not None:
        return B

    n = torch.arange(N, device=device, dtype=dtype)
    k = torch.arange(N, device=device, dtype=dtype)[:, None]
    B = torch.cos(math.pi * (n + 0.5) * k / N)
    # Orthonormal scaling
    B[0, :] /= math.sqrt(N)
    if N > 1:
        B[1:, :] *= math.sqrt(2.0 / N)

    _DCT_CACHE[key] = B
    return B


# ----------------------------------------------------------------------------------------------------------------------
def _apply_mat_along_axis(X: torch.Tensor, M: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Apply a square matrix M (N x N) along a specified axis of X.
    For each index of the remaining axes, perform y_axis = M @ x_axis.
    """
    assert M.shape[0] == M.shape[1], "M must be square"
    N = M.shape[0]
    assert X.shape[axis] == N, f"Matrix dimension {N} doesn't match X dim {X.shape[axis]} on axis {axis}"

    # Permute target axis to the front, apply matmul, and permute back.
    perm = [axis] + [i for i in range(X.ndim) if i != axis]
    Xp = X.permute(perm)                    # [N, ...]
    orig_shape = Xp.shape
    Xp = Xp.reshape(N, -1)                  # [N, rest]
    Yp = M @ Xp                             # [N, rest]
    Yp = Yp.reshape(orig_shape)

    # Inverse permutation
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return Yp.permute(inv)


# ======================================================================================================================
# DCT 2D (images) — rectangular low-pass selection or full coefficients
# ======================================================================================================================


# ======================================================================================================================
@register_encoder("dct_2d")
class DCT2DEncoder(BaseEncoder):
    """
    Encode an image (H, W) or (H, W, 1) using separable orthonormal 2D DCT-II.
    Keeping low-frequency blocks is handy for compression/de-noising.

    keep_h, keep_w:
        Keep only the top-left (keep_h x keep_w) low-frequency block; if None, keep full (H,W).
    keep_fraction:
        Alternative to keep_h/keep_w. If set, we keep:
            keep_h = ceil(H * sqrt(keep_fraction)), keep_w = ceil(W * sqrt(keep_fraction)).
        Ignored if explicit keep_h/keep_w are provided.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            keep_h: Optional[int] = None,
            keep_w: Optional[int] = None,
            keep_fraction: Optional[float] = None
    ):
        self.name = "dct_2d"
        self.keep_h = keep_h
        self.keep_w = keep_w
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_keeps(self, H: int, W: int) -> Tuple[int, int, str]:

        if self.keep_h is not None or self.keep_w is not None:
            kh = min(H, self.keep_h if self.keep_h is not None else H)
            kw = min(W, self.keep_w if self.keep_w is not None else W)
            return kh, kw, "rect"

        if self.keep_fraction is not None:
            s = max(0.0, min(1.0, float(self.keep_fraction)))
            side = math.sqrt(s)
            kh = max(1, min(H, int(math.ceil(H * side))))
            kw = max(1, min(W, int(math.ceil(W * side))))
            return kh, kw, "rect"

        # full keep
        return H, W, "full"

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:
        Y = _as_torch(signal, dtype=torch.float32)
        if Y.ndim == 3 and Y.shape[-1] == 1:
            Y2 = Y[..., 0]                   # (H,W)
            orig_3d = True
        elif Y.ndim == 2:
            Y2 = Y
            orig_3d = False
        else:
            raise AssertionError("DCT2DEncoder expects (H,W) or (H,W,1)")

        H, W = int(Y2.shape[0]), int(Y2.shape[1])
        device, dtype = Y2.device, Y2.dtype

        Bh = dct_matrix_1d(H, device=device, dtype=dtype)
        Bw = dct_matrix_1d(W, device=device, dtype=dtype)

        C = Bh @ Y2 @ Bw.t()                # (H,W) DCT coefficients

        kh, kw, mode = self._resolve_keeps(H, W)
        if mode == "full":
            emb = C.reshape(-1)             # (H*W,)
        else:
            emb = C[:kh, :kw].contiguous().reshape(-1)  # (kh*kw,)

        return {
            "embedding": emb,
            "shape": (H, W, 1 if orig_3d else 0),
            "keep_h": kh,
            "keep_w": kw,
            "mode": mode,
            "encoder_name": self.name,
            "transform": "dct2d_ortho",
        }

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, _ = input_shape
        kh, kw, mode = self._resolve_keeps(H, W)
        return kh * kw


# ======================================================================================================================
@register_decoder("dct_2d")
class DCT2DDecoder(BaseDecoder):
    """Inverse of DCT2DEncoder using stored keep_h/keep_w to reconstruct sparse/full spectrum."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            keep_h: Optional[int] = None,
            keep_w: Optional[int] = None,
            keep_fraction: Optional[float] = None
    ):
        self.name = "dct_2d"
        self.keep_h = keep_h
        self.keep_w = keep_w
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_encoder(cls, encoder: "DCT2DEncoder"):
        # Mirror encoder's keep policy so get_embedding_dim is accurate at build time
        return cls(keep_h=encoder.keep_h,
                   keep_w=encoder.keep_w,
                   keep_fraction=encoder.keep_fraction)

    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_keeps(self, H: int, W: int) -> Tuple[int, int]:
        if self.keep_h is not None or self.keep_w is not None:
            kh = min(H, self.keep_h if self.keep_h is not None else H)
            kw = min(W, self.keep_w if self.keep_w is not None else W)
            return kh, kw
        if self.keep_fraction is not None:
            s = max(0.0, min(1.0, float(self.keep_fraction)))
            side = math.sqrt(s)
            kh = max(1, min(H, int(math.ceil(H * side))))
            kw = max(1, min(W, int(math.ceil(W * side))))
            return kh, kw
        return H, W  # full

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, _ = input_shape
        kh, kw = self._resolve_keeps(H, W)
        return kh * kw

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:

        # Use encoder metadata for the inverse (authoritative).
        H, W, last_dim = metadata["shape"]
        kh = int(metadata["keep_h"])
        kw = int(metadata["keep_w"])
        mode = metadata.get("mode", "full")

        device, dtype = embedding.device, embedding.dtype
        Bh = dct_matrix_1d(H, device=device, dtype=dtype)
        Bw = dct_matrix_1d(W, device=device, dtype=dtype)

        if mode == "full":
            C = embedding.view(H, W)
        else:
            C = torch.zeros((H, W), device=device, dtype=dtype)
            C[:kh, :kw] = embedding.view(kh, kw)

        Y = Bh.t() @ C @ Bw
        if last_dim == 1:
            Y = Y.unsqueeze(-1)

        return Y


# ======================================================================================================================
# DCT 3D (videos) — separable 3D DCT-II with rectangular low-pass or full
# ======================================================================================================================

# ======================================================================================================================
@register_encoder("dct_3d")
class DCT3DEncoder(BaseEncoder):
    """
    Encode a video (H, W, T) using separable orthonormal 3D DCT-II.
    keep_h/keep_w/keep_t select a low-frequency block; keep_fraction uses cubic scaling s^(1/3).
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            keep_h: Optional[int] = None,
            keep_w: Optional[int] = None,
            keep_t: Optional[int] = None,
            keep_fraction: Optional[float] = None
    ):
        self.name = "dct_3d"
        self.keep_h = keep_h
        self.keep_w = keep_w
        self.keep_t = keep_t
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_keeps(self, H: int, W: int, T: int) -> Tuple[int, int, int, str]:
        if self.keep_h is not None or self.keep_w is not None or self.keep_t is not None:
            kh = min(H, self.keep_h if self.keep_h is not None else H)
            kw = min(W, self.keep_w if self.keep_w is not None else W)
            kt = min(T, self.keep_t if self.keep_t is not None else T)
            return kh, kw, kt, "rect"
        if self.keep_fraction is not None:
            s = max(0.0, min(1.0, float(self.keep_fraction))) ** (1.0 / 3.0)
            kh = max(1, min(H, int(math.ceil(H * s))))
            kw = max(1, min(W, int(math.ceil(W * s))))
            kt = max(1, min(T, int(math.ceil(T * s))))
            return kh, kw, kt, "rect"
        return H, W, T, "full"

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, signal) -> Dict[str, Any]:

        X = _as_torch(signal, dtype=torch.float32)
        assert X.ndim == 3, "DCT3DEncoder expects (H, W, T)"
        H, W, T = map(int, X.shape)
        device, dtype = X.device, X.dtype

        Bh = dct_matrix_1d(H, device=device, dtype=dtype)
        Bw = dct_matrix_1d(W, device=device, dtype=dtype)
        Bt = dct_matrix_1d(T, device=device, dtype=dtype)

        # Forward 3D DCT: along H, then W, then T
        C = _apply_mat_along_axis(X, Bh, axis=0)   # H dimension
        C = _apply_mat_along_axis(C, Bw, axis=1)   # W dimension
        C = _apply_mat_along_axis(C, Bt, axis=2)   # T dimension

        kh, kw, kt, mode = self._resolve_keeps(H, W, T)
        if mode == "full":
            emb = C.reshape(-1)                    # (H*W*T,)
        else:
            emb = C[:kh, :kw, :kt].contiguous().reshape(-1)  # (kh*kw*kt,)

        return {
            "embedding": emb,
            "shape": (H, W, T),
            "keep_h": kh,
            "keep_w": kw,
            "keep_t": kt,
            "mode": mode,
            "encoder_name": self.name,
            "transform": "dct3d_ortho",
        }

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, T = input_shape
        kh, kw, kt, _ = self._resolve_keeps(H, W, T)
        return kh * kw * kt


# ======================================================================================================================
@register_decoder("dct_3d")
class DCT3DDecoder(BaseDecoder):
    """Inverse of DCT3DEncoder; reconstructs full/sparse spectrum then applies inverse along T, W, H."""

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
            self,
            keep_h: Optional[int] = None,
            keep_w: Optional[int] = None,
            keep_t: Optional[int] = None,
            keep_fraction: Optional[float] = None
    ):
        self.name = "dct_3d"
        self.keep_h = keep_h
        self.keep_w = keep_w
        self.keep_t = keep_t
        self.keep_fraction = keep_fraction

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_encoder(cls, encoder: "DCT3DEncoder"):
        # Mirror encoder's keep policy so get_embedding_dim is accurate at build time
        return cls(keep_h=encoder.keep_h,
                   keep_w=encoder.keep_w,
                   keep_t=encoder.keep_t,
                   keep_fraction=encoder.keep_fraction)

    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_keeps(self, H: int, W: int, T: int) -> Tuple[int, int, int]:
        if self.keep_h is not None or self.keep_w is not None or self.keep_t is not None:
            kh = min(H, self.keep_h if self.keep_h is not None else H)
            kw = min(W, self.keep_w if self.keep_w is not None else W)
            kt = min(T, self.keep_t if self.keep_t is not None else T)
            return kh, kw, kt
        if self.keep_fraction is not None:
            s = max(0.0, min(1.0, float(self.keep_fraction))) ** (1.0 / 3.0)
            kh = max(1, min(H, int(math.ceil(H * s))))
            kw = max(1, min(W, int(math.ceil(W * s))))
            kt = max(1, min(T, int(math.ceil(T * s))))
            return kh, kw, kt
        return H, W, T  # full

    # ------------------------------------------------------------------------------------------------------------------
    def get_embedding_dim(self, input_shape: Tuple[int, int, int]) -> int:
        H, W, T = input_shape
        kh, kw, kt = self._resolve_keeps(H, W, T)
        return kh * kw * kt

    # ------------------------------------------------------------------------------------------------------------------
    def decode_torch(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:

        # Use encoder metadata for the inverse (authoritative).
        H, W, T = metadata["shape"]
        kh = int(metadata["keep_h"])
        kw = int(metadata["keep_w"])
        kt = int(metadata["keep_t"])
        mode = metadata.get("mode", "full")

        device, dtype = embedding.device, embedding.dtype
        Bh = dct_matrix_1d(H, device=device, dtype=dtype)
        Bw = dct_matrix_1d(W, device=device, dtype=dtype)
        Bt = dct_matrix_1d(T, device=device, dtype=dtype)

        if mode == "full":
            C = embedding.view(H, W, T)
        else:
            C = torch.zeros((H, W, T), device=device, dtype=dtype)
            C[:kh, :kw, :kt] = embedding.view(kh, kw, kt)

        # Inverse along T, then W, then H (note: inverse is transpose for orthonormal DCT-II)
        Y = _apply_mat_along_axis(C, Bt.t(), axis=2)
        Y = _apply_mat_along_axis(Y, Bw.t(), axis=1)
        Y = _apply_mat_along_axis(Y, Bh.t(), axis=0)

        return Y


# ======================================================================================================================
if __name__ == "__main__":

    # ==================================================================================================================
    # KEEP: Synthetic tests for quick encoder/decoder sanity checks
    # ------------------------------------------------------------------------------------------------------------------
    # The following block is extremely useful when developing or reviewing a new encoder/decoder: generate synthetic
    # data, round-trip encode/decode, compute an error metric (e.g., MSE), and visualize.
    # ==================================================================================================================

    TESTS_TO_RUN = {
        "time_series": False,
        "profiles": False,
        "images": False,
        "videos": False
    }

    # ------------------------------------------------------------------------------------------------------------------
    # TIME SERIES

    if TESTS_TO_RUN["time_series"]:

        # Settings
        TS_ENCODING = "BSpline_flat"  # Options: "BSpline_flat"
        T_ = 100
        grid = np.arange(T_)
        signal_ = np.sin(2 * np.pi * grid / T_) + 0.05 * np.random.randn(T_)

        # Encode and decode

        if TS_ENCODING == "BSpline_flat":
            # BSplines flatten
            encoder_ = FlattenBsplineEncoder1D(num_basis=10, degree=3)
            decoder_ = FlattenBsplineDecoder1D(degree=3)
        else:
            raise ValueError(f"Invalid TS_ENCODING {TS_ENCODING}. Valid options: 'BSpline_flat'.")

        encoded = encoder_.encode(signal_)
        embeddings = encoded['embedding']
        print(f'Embedding shape: {embeddings.shape}')
        reconstructed = decoder_.decode(encoded["embedding"], encoded)

        # Evaluate reconstruction
        mse = mean_squared_error(signal_, reconstructed)

        # Visualize

        plt.figure(figsize=(10, 4))
        plt.plot(grid, signal_, label="Original Signal", linewidth=2)  # noqa
        plt.plot(grid, reconstructed, '--', label="Reconstructed Signal", linewidth=2)  # noqa
        plt.title(f"Flatten B-spline 1D Reconstruction (MSE = {mse:.4e})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PROFILES

    if TESTS_TO_RUN["profiles"]:

        # Settings
        PROFILE_ENCODING = "BSpline_chan"  # Options: "BSpline_flat", "BSpline_chan", "FPC"
        C_, T_ = 10, 20
        signal_2d = np.stack([  # noqa
            np.sin(2 * np.pi * np.linspace(0, 1, T_) + phase) + 0.05 * np.random.randn(T_)
            for phase in np.linspace(0, np.pi, C_)
        ], axis=0).reshape(C_, 1, T_)  # Shape: (C_, 1, T_)

        # Encode and decode

        if PROFILE_ENCODING == "BSpline_flat":
            # BSplines flatten
            encoder_ = FlattenBsplineEncoder1D(num_basis=60, degree=3)
            decoder_ = FlattenBsplineDecoder1D(degree=3)
        elif PROFILE_ENCODING == "BSpline_chan":
            # BSplines per channel
            encoder_ = PerChannelBsplineEncoder1D(degree=3, num_basis=4)
            decoder_ = PerChannelBsplineDecoder1D(degree=3)
        elif PROFILE_ENCODING == "FPC":
            # FPC encoder/decoder
            encoder_ = FPCA3DEncoder(num_components=2, compression=None, pca_dim="space")
            decoder_ = FPCA3DDecoder()
        else:
            raise ValueError(
                f"Invalid PROFILE_ENCODING {PROFILE_ENCODING}. Valid options: 'BSpline_flat', 'BSpline_flat', 'FPC'."
            )

        encoded = encoder_.encode(signal_2d)
        embeddings = encoded['embedding']
        print(f'Embedding shape: {embeddings.shape}')
        reconstructed = decoder_.decode(encoded["embedding"], encoded)

        # Visualize

        C_ = signal_2d.shape[0]
        T_ = signal_2d.shape[-1]

        fig, axs = plt.subplots(C_, 1, figsize=(10, 2.5 * C_), sharex=True)

        for c in range(C_):
            axs[c].plot(signal_2d[c, 0], label="Original", linewidth=2)
            axs[c].plot(reconstructed[c, 0], "--", label="Reconstructed", linewidth=2)
            axs[c].set_title(f"Channel {c}")
            axs[c].legend()
            axs[c].grid(True)

        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # IMAGES

    if TESTS_TO_RUN["images"]:

        # Settings
        IMAGE_ENCODING = "BSpline_flat"  # Options: "DCT", "BSpline_flat", "FPC"
        H_, W_, T_ = 64, 64, 1
        grid = np.arange(T_)

        # Generate structured "video" with one frame (image)
        video = generate_single_channel_video(height=H_, width=W_, num_frames=T_)

        # Encode and decode

        if IMAGE_ENCODING == "DCT":
            # DCT encoder/decoder
            encoder_ = DCT2DEncoder(keep_h=8, keep_w=8)
            decoder_ = DCT2DDecoder()
        elif IMAGE_ENCODING == "BSpline_flat":  # ROH
            # Encode/Decode with flatten-based B-spline
            encoder_ = FlattenBsplineEncoder1D(num_basis=1000, degree=3)
            decoder_ = FlattenBsplineDecoder1D(degree=3)
        elif IMAGE_ENCODING == "FPC":  # ROH
            # FPC encoder/decoder
            encoder_ = FPCA2DEncoder(num_components=2, pca_dim="space")
            decoder_ = FPCA2DDecoder()
        else:
            raise ValueError(f"Invalid IMAGE_ENCODING {IMAGE_ENCODING}. Valid options: 'DCT', 'BSpline_flat', 'FPC'.")

        encoded = encoder_.encode(video)
        embeddings = encoded['embedding']
        print(f'Embedding shape: {embeddings.shape}')
        reconstructed_video = decoder_.decode(embeddings, encoded)

        # Visualize

        frame_idx = np.arange(T_)
        for frame in frame_idx:  # noqa

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(video[:, :, frame], cmap="viridis")
            plt.title("Original Frame")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_video[:, :, frame], cmap="viridis")
            plt.title("Reconstructed Frame")

            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # VIDEOS
    # REMARKS:
    # - BSpline_flat does not work for videos.

    if TESTS_TO_RUN["videos"]:

        # Settings
        VIDEO_ENCODING = "FPC"  # Options: "FPC", "DCT"
        H_, W_, T_ = 64, 64, 1  # 5
        grid = np.arange(T_)

        # Generate structured video with spatial phase shift
        video = generate_single_channel_video(height=H_, width=W_, num_frames=T_)

        # Encode and decode

        if VIDEO_ENCODING == "FPC":
            # FPC encoder/decoder
            encoder_ = FPCA3DEncoder(num_components=5, compression=None, pca_dim="space")
            decoder_ = FPCA3DDecoder()
        elif VIDEO_ENCODING == "DCT":
            # DCT encoder/decoder
            encoder_ = DCT3DEncoder(keep_h=10, keep_w=10, keep_t=5)
            decoder_ = DCT3DDecoder()
        else:
            raise ValueError(f"Invalid VIDEO_ENCODING {VIDEO_ENCODING}. Valid options: 'FPC', 'DCT'.")

        encoded = encoder_.encode(video)
        embeddings = encoded['embedding']
        print(f'Embedding shape: {embeddings.shape}')
        reconstructed_video = decoder_.decode(embeddings, encoded)

        # Visualize

        frame_idx = np.arange(T_)
        for frame in frame_idx:  # noqa

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(video[:, :, frame], cmap="viridis")
            plt.title("Original Frame")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_video[:, :, frame], cmap="viridis")
            plt.title("Reconstructed Frame")

            plt.tight_layout()
            plt.show()
