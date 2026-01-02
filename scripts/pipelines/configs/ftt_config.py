# ftt_config.py
"""
FT-Transformer config
=====================

This file centralizes all knobs used by the FT-Transformer pipeline:
data sources, windowing, encoders/decoders, training flags, and device/dtype.

How encoders/decoders are chosen
--------------------------------
You can specify an encoder *per variable* directly in the specs:

    INPUT_SPECS["pf_active-coil_current"] = InputSpec(
        name="pf_active-coil_current",
        encoder_name="flatten_bspline_1d",
        encoder_kwargs={"degree": 4, "num_basis": 5},
    )

    TARGET_SPECS["equilibrium-elongation"] = TargetSpec(
        name="equilibrium-elongation",
        encoder_name="per_channel_bspline_1d",
        encoder_kwargs={"degree": 4, "num_basis": 5},
        head_hidden=128,
        loss="mse",
        loss_weight=1.0,
    )

If an encoder is **not** specified in the spec, the pipeline will:
1) Infer each variable’s **modality** from its **bound shape** (after the first shot is processed),
2) Look up a default in `DEFAULT_INPUT_ENCODERS_BY_MOD` (for inputs) or
   `DEFAULT_TARGET_ENCODERS_BY_MOD` (for targets),
3) Fill the encoder fields accordingly.

- Targets: the decoder is **auto-derived** from the target’s encoder via the registry
  (when an inverse exists, e.g., DCT/FPCA/B-spline). If no inverse exists, decoder stays `None`.
- Inputs: `encoder=None` is supported (FTTransformPrep normalizes shapes to 3D).

Parameter blocks (what they mean)
---------------------------------
• Repro & verbosity
  - `SEED`: global RNG seed used where applicable.
  - `VERBOSE`: print more diagnostics from the pipeline/model.

• General settings
  - `LOCAL_FLAG`: dataset location toggle used by `MastDataset`.
  - `SUBSET_OF_SHOTS`: limit dataset size (e.g., 1 for quick runs, `None` for full set).
  - `NUM_WORKERS`: DataLoader workers.
  - `OUTPUT_SUB_FOLDER`: subdir under `output/` for artifacts/checkpoints.
  - `SAVE_RESULTS`: if True, write evaluation CSVs/checkpoints.
  - `RUN_EVALUATION`, `RUN_TRAINING`: enable/disable those phases.

• Signal list
  - `SOURCE_SIGNAL_LIST`: list of `(source, signal)` to pull from MAST. Keys used in specs are
    `"{source}-{signal}"`.

• Specs (per-variable)
  - `INPUT_SPECS`: variables used as **inputs**. Shapes are *bound later* from data.
    You may set `encoder_name/encoder_kwargs` here; otherwise defaults by modality apply.
  - `TARGET_SPECS`: variables used as **targets**. Same rule: optional explicit encoder here,
    otherwise defaults by modality apply.
  - `INACTIVE_TARGETS`: target names to exclude from training/eval.

• Per-modality encoder defaults
  - `DEFAULT_INPUT_ENCODERS_BY_MOD`: fallback encoders for inputs if a spec didn’t set one.
  - `DEFAULT_TARGET_ENCODERS_BY_MOD`: fallback encoders for targets (decoder inferred when possible).
  - Modalities include: "scalar", "vector", "timeseries", "profile", "image", "video".
    (If a modality has no mapping here and the spec didn’t set an encoder, encoder remains `None`.)

• Windowing / transforms
  - `WINDOW_SEGMENTER_PARAMS`: parameters for `WindowSegmenterTransform`. `x_keys` and `y_keys`
    are derived from the spec dictionaries. Windowing directly affects the bound shapes (and thus
    the inferred modality and default encoders). FTTransformPrep then ensures **3D** shapes for both X and Y.

• Training
  - `BATCH_SIZE`, `EPOCHS`, `LR_TRUNK`, `LR_HEADS`, `USE_ADAMW`, `EARLY_STOP_PATIENCE`.
  - `LOSS_SPACE`: "pred" (compute MSE on head outputs) or "native" (decode first, then MSE in native space).

• Device & dtype
  - `MODEL_DTYPE`: float16 on CUDA by default, else float32.
  - `DEVICE`: picks cuda/mps/cpu in that order.

Implementation note
-------------------
The pipeline binds shapes from the first train shot (no DataLoader spin-up) and calls
`build_registries_from_shapes(...)` to create `InputRegistry` and `TargetRegistry`, applying the
per-modality defaults where needed. FTTransformPrep makes both X and Y 3D so that `encoder=None`
works cleanly for inputs.
"""

from __future__ import annotations
# import torch
from scripts.pipelines.models.ftt_model import InputSpec, TargetSpec

# ======================================================================================================================
# Reproducibility & verbosity
# ======================================================================================================================

SEED = 54
VERBOSE = True

# ======================================================================================================================
# General settings
# ======================================================================================================================

LOCAL_FLAG = False  # True
SUBSET_OF_SHOTS = 4
NUM_WORKERS = 0  # 64
OUTPUT_SUB_FOLDER = 'ftt_output/'
SAVE_RESULTS = True
RUN_EVALUATION = True
RUN_TRAINING = True

# ======================================================================================================================
# SIGNAL LIST
# ======================================================================================================================

SOURCE_SIGNAL_LIST = [
    ('magnetics', 'flux_loop_flux'),
    ('magnetics', 'b_field_pol_probe_ccbv_field'),
    ('magnetics', 'b_field_pol_probe_obr_field'),
    ('magnetics', 'b_field_pol_probe_obv_field'),
    ('pf_active', 'solenoid_current'),
    ('pf_active', 'coil_voltage'),
    ('pf_active', 'coil_current'),
    ('pulse_schedule', 'i_plasma'),
    ('summary', 'power_nbi'),
    ('equilibrium', 'elongation'),
    ('equilibrium', 'elongation_axis'),
    ('equilibrium', 'triangularity_upper'),
    ('equilibrium', 'triangularity_lower'),
    ('equilibrium', 'minor_radius'),
    ('equilibrium', 'magnetic_axis_r'),
    ('equilibrium', 'magnetic_axis_z')
]

# ======================================================================================================================
# INPUT specs (no shapes; keys define x_keys)
# Encoders are assigned later via DEFAULT_INPUT_ENCODERS_BY_MOD if not set here.
# ======================================================================================================================

INPUT_SPECS = {
    # Time series & profiles you’re using as inputs
    "magnetics-flux_loop_flux": InputSpec(
        name="magnetics-flux_loop_flux",
        # encoder_name="flatten_bspline_1d",
        # encoder_kwargs={"degree": 3, "num_basis": 5}
    ),
    "magnetics-b_field_pol_probe_ccbv_field": InputSpec(
        name="magnetics-b_field_pol_probe_ccbv_field"
    ),
    "magnetics-b_field_pol_probe_obr_field": InputSpec(
        name="magnetics-b_field_pol_probe_obr_field"
    ),
    # "magnetics-b_field_pol_probe_obv_field": InputSpec(
    #     name="magnetics-b_field_pol_probe_obv_field"
    # ),
    # "pf_active-solenoid_current": InputSpec(
    #     name="pf_active-solenoid_current"
    # ),
    # "pf_active-coil_voltage": InputSpec(
    #     name="pf_active-coil_voltage"
    # ),
    # "pf_active-coil_current": InputSpec(
    #     name="pf_active-coil_current"
    # ),
    # "pulse_schedule-i_plasma": InputSpec(
    #     name="pulse_schedule-i_plasma"
    # ),
    # "summary-power_nbi": InputSpec(
    #     name="summary-power_nbi"),
}

# ======================================================================================================================
# TARGET specs (no shapes; keys define y_keys)
# Encoders are assigned later using DEFAULT_TARGET_ENCODER
# ======================================================================================================================

TARGET_SPECS = {
    "equilibrium-elongation": TargetSpec(
        name="equilibrium-elongation",
        # encoder_name="flatten_bspline_1d",
        # encoder_kwargs={"degree": 3, "num_basis": 5}
        head_hidden=128,
        loss="mse",
        loss_weight=1.0
    ),
    "equilibrium-elongation_axis": TargetSpec(
        name="equilibrium-elongation_axis",
        head_hidden=128,
        loss="mse",
        loss_weight=1.0
    ),
    # "equilibrium-triangularity_upper": TargetSpec(
    #     name="equilibrium-triangularity_upper",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-triangularity_lower": TargetSpec(
    #     name="equilibrium-triangularity_lower",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-minor_radius": TargetSpec(
    #     name="equilibrium-minor_radius",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
    # "equilibrium-magnetic_axis_r": TargetSpec(
    #     name="equilibrium-magnetic_axis_r",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0)
    # ,
    # "equilibrium-magnetic_axis_z": TargetSpec(
    #     name="equilibrium-magnetic_axis_z",
    #     head_hidden=128,
    #     loss="mse",
    #     loss_weight=1.0
    # ),
}

# Optional: exclude some targets from training/eval
INACTIVE_TARGETS: list[str] = []

# ======================================================================================================================
# Modality-based encoder defaults
# Applied AFTER shapes are bound from the first shot
# ======================================================================================================================

DEFAULT_INPUT_ENCODERS_BY_MOD = {
    # "timeseries": dict(encoder_name="flatten_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    "profile": dict(encoder_name="fpca_3d", encoder_kwargs={"num_components": 5, "pca_dim": "space"}),
    # "profile": dict(encoder_name="per_channel_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    # "image":   dict(encoder_name="dct_2d", encoder_kwargs={"keep_h": 8, "keep_w": 8}),
    "video":   dict(encoder_name="dct_3d", encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}),
    "scalar":  dict(encoder_name=None, encoder_kwargs=None),
    "vector":  dict(encoder_name=None, encoder_kwargs=None),
}

DEFAULT_TARGET_ENCODERS_BY_MOD = {
    "timeseries": dict(encoder_name="flatten_bspline_1d",     encoder_kwargs={"degree": 4, "num_basis": 5}),
    "profile":    dict(encoder_name="per_channel_bspline_1d", encoder_kwargs={"degree": 4, "num_basis": 5}),
    "image":      dict(encoder_name="dct_2d",                 encoder_kwargs={"keep_h": 8, "keep_w": 8}),
    "video":      dict(encoder_name="dct_3d",                 encoder_kwargs={"keep_h": 5, "keep_w": 5, "keep_t": 5}),
    "scalar":     dict(encoder_name=None, encoder_kwargs=None),
    "vector":     dict(encoder_name=None, encoder_kwargs=None),
}

# ======================================================================================================================
# Transforms parameters
# x_keys and y_keys are derived from the specs above
# ======================================================================================================================

WINDOW_SEGMENTER_PARAMS = {
    "x_keys": list(INPUT_SPECS.keys()),   # auto from INPUT_SPECS
    "y_keys": list(TARGET_SPECS.keys()),  # auto from TARGET_SPECS
    "x_window_sec": 0.01,
    "y_window_sec": 0.1,
    "dt_sec": 0.025,
    "stride_sec": None,
    "stride_unitary": True,
    "min_samples_per_window": 1,
    "verbose": False,
}

REF_FREQ = 0.005  # kept for compatibility; currently unused if sampling transform is commented

# ======================================================================================================================
# Training params
# ======================================================================================================================

BATCH_SIZE = 100
EPOCHS = 3
LR_TRUNK = 1e-3
LR_HEADS = 1e-3
USE_ADAMW = False
LOSS_SPACE = "native"  # 'pred' or 'native'
EARLY_STOP_PATIENCE = 5

# ======================================================================================================================
