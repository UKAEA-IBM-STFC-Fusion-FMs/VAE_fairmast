
# `vae_pipeline.py` — MAST β-VAE training script

## Getting started
git clone --recurse-submodules git@gitlab.stfc.ac.uk:hncdi-fusion-plasma-modelling/VAE_fairmast.git

### Create conda environment
1- Install Miniforge distribution compatible with your operative system.
2- Create conda environment:

``` 
conda create -n VAEfairmast python=3.11
```

3- Activate environment:

```
conda activate VAEfairmast
```

### Install dependencies (tokamark)
```
pip install -e tokamark
```

**Dataset:**  
The pipeline operates on the *tokamark* dataset, available at:  
https://huggingface.co/datasets/UKAEA-IBM-STFC/tokamark-dataset


**Path:** `src/vae_pipeline/vae_pipeline.py`

`vae_pipeline.py` is the CLI entry point that trains a single **β-VAE** to learn a compact
latent representation of a MAST diagnostic signal from the **TokaMark dataset**.
The entire run is driven by two files:

- a **JSON config** (`--config_file_path`) — data, model architecture, and training hyperparameters;
- a **YAML task config** (`--config_task_file_path`) — TokaMark task/windowing metadata.

A new feature of this version is the **masked loss**: the reconstruction term is computed only
over finite (valid) entries of the signal, so NaNs no longer pollute training.

## Usage

```bash
python src/vae_pipeline/vae_pipeline.py \
  --config_file_path path/to/your/config.json \
  --config_task_file_path path/to/your/config.yaml
```

## What it does, step by step

### 1. Setup (`main()`)

1. Picks the device: **CUDA** if available, otherwise CPU.
2. Parses the two config arguments.
3. Loads the JSON config into a `Settings` object via
   `get_settings()` (`src/vae_pipeline/configs/config_setup.py`).
4. Loads the YAML task config and builds TokaMark task metadata
   (`tokamark.tasks.get_task_metadata`).
5. Creates the output directory
   `SETTINGS.LOCAL_PATHS.data_output_directory + "conv1d_vae_<config_name>/"`.
6. Selects train/validation **shot IDs** from the data-split CSV
   (`get_train_test_val_shots`); shot `24623` is removed from validation.
7. Loads per-signal mean/std statistics from `dict_signals_stats.yaml`.
8. Builds a per-signal transform map (for instance):
   - `StdScalingTransform` (z-scoring with outlier cleaning) for every signal, plus
9. Builds the raw datasets (`MastDataset` via `initialize_datasets`) over the local
   zarr store, then wraps them with TokaMark's `initialize_TokaMark_dataset` and a
   `ModelSpecificTransform` that produces the `batch["x"]` tensor.
10. Creates `DataLoader`s (batch size / workers from config).
11. Instantiates the model `beta_VAE(SETTINGS)` (see below).
12. Creates an **Adam** optimizer and a
    **`CosineAnnealingWarmRestarts`** LR scheduler.
13. Saves `model.json` (architecture) and a copy of the JSON config into the output dir.
14. Calls `train_vae_model(...)` and prints elapsed time.

### 2. Model (`beta_VAE`, `src/vae_pipeline/models/vae_model.py`)

- **Encoder/decoder**: built by `EncoderDecoder`
  (`src/vae_pipeline/models/encoder_decoder.py`) — a `conv1d` stack, a `linear` stack, or a
  generic config-driven `"encoder_decoder"` stack, depending on the config.
- **Latent head**: two linear layers produce `mu` and `logvar` of the posterior
  `q(z|x)`; a reparameterized sample `z` is decoded back to signal space.
- **Input preparation**: a binary validity mask is built from finite entries, NaNs are
  imputed with 0, and `[imputed_x, mask]` are concatenated along the channel dimension,
  so the network *sees* where data is missing. Handles 1D (conv1d/linear) and 2D (conv2d)
  layouts.
- **Loss** (`masked_loss_function`):
  `total = masked_MSE + beta * KL`, where the MSE is masked to valid entries and averaged
  per-sample, and `logvar` is clamped for numerical stability.

### 3. Training loop (`train_vae_model`)

Per epoch:

1. **Train phase** — for each batch:
   - skip the batch if >25% of samples are <75% finite;
   - forward + masked loss via `training_block`
     (`src/vae_pipeline/utils/utils.py`) under **AMP** (autocast + `GradScaler` on CUDA);
   - skip batches with non-finite loss components;
   - gradient clipping (`max_norm=1`), skipping the step if the grad norm is NaN/Inf;
   - optimizer step (AMP-aware).
2. **Validation phase** — same forward + masked loss under `no_grad`, averaged over the set.
3. **Adaptive β schedule** — KL and reconstruction losses are EMA-smoothed (decay 0.7);
   after `patience` warm-up epochs, β is multiplicatively updated to keep
   `KL/recon ≈ target_scale (0.1)`, with the step clipped to ±0.2 in log space and β
   bounded to `[1e-5, 10]`.
4. **Checkpointing** — on a new best validation loss, `best_vae_<signal>.pt` is saved
   (model/optimizer/scheduler state + epoch); `last_vae_<signal>.pt` is saved every epoch;
   `loss_curves.json` stores train/val loss curves plus LR and β histories.
5. **Early stopping** — if validation loss does not improve for `patience` epochs and
   the `min_nr_epochs` floor has passed, training stops.

### 4. Outputs (in the output directory)

| File | Content |
| --- | --- |
| `best_vae_<signal>.pt` | best checkpoint (model/optimizer/scheduler state + epoch) |
| `last_vae_<signal>.pt` | last checkpoint |
| `loss_curves.json` | train/val total-recon-KL curves, LR and β history |
| `model.json` | model architecture string |
| `<config>.json` | copy of the config used for the run |

## Dependencies

The script cannot run in isolation; it depends on the following.

### Python / third-party packages

- **`torch`** (PyTorch, >= 2.x) — the core: model definition, training loop, AMP
  (`torch.amp.autocast`/`GradScaler`), `DataLoader`, Adam, LR schedulers, checkpoint saving.
- **`numpy`** — β adaptive schedule arithmetic.
- **`PyYAML` (`yaml`)** — loading the task config (via TokaMark) and the
  `dict_signals_stats.yaml` signal statistics.
- **`pandas`** (indirect, via `src/utils/utils.py`) — reading the shot-ID split CSV.
- Conda env per `environment.yml` (Python 3.11): `zarr`, `fsspec`, `s3fs`, `xarray`,
  `tqdm`, `rich`, etc. (used by the data stack below).

### In-repo modules

- `src.vae_pipeline.configs.config_setup` — `get_settings()` (JSON → `Settings`).
- `src.vae_pipeline.models.vae_model` — `beta_VAE` and `masked_loss_function`.
- `src.vae_pipeline.models.encoder_decoder` (+ `encoder_decoder_utils`) — encoder/decoder
  builders (`conv1d`, `linear`, config-driven stacks).
- `src.vae_pipeline.utils.utils` — `training_block` (one forward + masked loss step).
- `src.common_transforms.general_transforms` — `ModelSpecificTransform`,
  `StdScalingTransform`, `CropSignalFeatures`.
- `src.utils.utils` — `ComposeTransforms`, `initialize_datasets`,
  `get_train_test_val_shots`, `load_task_config`.

### External packages 

- **`MAST_tools`** — `MastDataset`, `CachedDataset`: reads raw MAST diagnostic data from
  the zarr store and applies the per-signal transforms.
- **`tokamark`** — `get_task_metadata`, `initialize_TokaMark_dataset`,
  `ReshapeLcfsTransform`, and `get_config_from_yaml` (YAML task loading). If the repo's
  `tokamark/` directory is an empty placeholder; install TokaMark separately
  (e.g. `pip install -e tokamark` as in the README).

### Data & environment dependencies

- **TokaMark dataset** (zarr) at the hard-coded local path
  `/lustre/home/bf3280/tokamark_fairmast_dataset` (used when `"local": true` in the
  config; this path is machine-specific and must exist).
- **Data-split CSV** at `SETTINGS.LOCAL_PATHS.data_split_csv_path`
  (`TokaMark_data_splits.csv`) — defines train/val/test shot IDs.
- **Signal statistics** at `SETTINGS.LOCAL_PATHS.global_mean_std_path/dict_signals_stats.yaml`
  (per-signal mean/std for z-scoring).
- The **JSON config file** and **YAML task file** passed on the command line.
- A **CUDA GPU** is optional: training works on CPU, but AMP is only enabled on CUDA.


