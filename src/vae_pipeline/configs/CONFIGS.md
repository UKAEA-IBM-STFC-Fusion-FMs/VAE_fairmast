# VAE Pipeline Configuration Reference

Field-by-field documentation of the VAE pipeline JSON configs
(`src/vae_pipeline/configs/config_*.json`), based on how each field is
consumed in the code. The config is parsed by `Settings`
(`src/vae_pipeline/configs/config_setup.py:26`) and passed as `SETTINGS`
to the training pipeline (`vae_pipeline.py`), the visualization script
(`vae_pipeline_visualization.py`) and the model builders
(`models/vae_model.py`, `models/encoder_decoder*.py`).

One config describes **one VAE for one signal** (see `input.data_names`).
Run with:

```
python src/vae_pipeline/vae_pipeline.py --config_file_path <config.json> --config_task_file_path src/vae_pipeline/configs/task_encoding_VAE.yaml
```

## Top-level sections

| Section | Required | Parsed as | Purpose |
|---|---|---|---|
| `input` | yes (KeyError if absent) | `SETTINGS.DATA` | Which signal(s) the VAE encodes. See [input](#input). |
| `time_settings` | yes | `SETTINGS.TIME_SEGMENTATION` | Time-windowing parameters. See [time_settings](#time_settings). |
| `windowed_data_specs` | yes | `SETTINGS.WINDOWsSHAPE` | Window shape (channels × length). See [windowed_data_specs](#windowed_data_specs). |
| `training` | yes | `SETTINGS.TRAINING` | Training hyper-parameters. See [training](#training). |
| `beta-vae` | yes | `SETTINGS.BETA_VAE` | Latent dimension and β-VAE weights. See [beta-vae](#beta-vae). |
| `encoder` | yes | `SETTINGS.ENCODER` | Encoder type and layer specs. See [encoder](#encoder). |
| `paths` | yes | `SETTINGS.LOCAL_PATHS` | Data/output locations. See [paths](#paths). |
| `decoder` | no (but see [encoder](#encoder)) | `SETTINGS.DECODER` | Explicit decoder layer specs. See [decoder](#decoder). |
| `conv1d_encoder` | no | `SETTINGS.CONV1dENCODER` | Legacy single-conv spec; validation only. See [conv1d_encoder](#conv1d_encoder). |
| `scheduler` | no | `SETTINGS.SCHEDULER` | LR-scheduler spec; parsed but unused. See [scheduler](#scheduler). |
| `local` | no (default `True`) | `SETTINGS.DATA.local` | Local zarr store vs remote MAST. See [local](#local). |
| `cache_data` | no (default `True`) | `SETTINGS.DATA.cache_data` | Dataset caching flag. See [cache_data](#cache_data). |

Within a section, missing keys print a `[Warning]` and are set to
`None`; model construction then fails the checks in
`_chek_for_missing_attributes` (`models/encoder_decoder_utils.py:435`).

---

## `input`

Parsed by `DataInput` (`config_setup.py:228`).

| Field | Type | Role in code |
|---|---|---|
| `data_names` | list[[source, signal]] | The signal(s) this VAE compresses. Used as the dataset `source_signal_list` (`vae_pipeline.py:402,454`) and the **first entry's signal name** sets the checkpoint file names `best_vae_{signal_name}.pt` / `last_vae_{signal_name}.pt` (`vae_pipeline.py:62,306,319`) — the same names the benchmark pipeline later loads (`src/benchmark/utils_new.py:74`). The pipeline is single-VAE-per-config in practice. |
| `target_names` | list[[source, signal]] | Reconstruction target signals. Combined with `data_names` into `all_source_signal_list` (`config_setup.py:243-247`). For the self-supervised VAE training used here it is identical to `data_names` (as in `config_flux_loop_flux.json`); the pipeline itself only loads data for `data_names`. |

---

## `time_settings`

Parsed by `TimeSettings` (`config_setup.py:154`). These are the same
parameters as the windowing transform
(`WindowSegmenterTransform`,
`src/common_transforms/window_segmenter_transform.py`) used by the task
YAML (`task_encoding_VAE.yaml`); the copy in the VAE config keeps the
model's expected input shape consistent with the dataset windowing.

| Field | Type | Role in code |
|---|---|---|
| `x_window_sec` | float | Duration (s) of the input time window. `0` → single sample (max Δt of the x signals) (`window_segmenter_transform.py:196-207`). |
| `y_window_sec` | float | Duration (s) of the target time window. `0` → single sample. For auto-encoding VAEs this equals `x_window_sec`. |
| `dt_sec` | float | Forecasting delay (s) between the end of the x-window and the start of the y-window (`window_segmenter_transform.py:251,284`). `0` for pure reconstruction. |
| `stride_sec` | float | Time (s) between the starts of consecutive windows. If `stride_unitary` is true the effective stride is `max(stride_sec, max native Δt)` (`window_segmenter_transform.py:219-226`). |
| `stride_unitary` | bool | If `true`, the stride is set to the maximum native sampling interval across all signals so every signal (even at different rates) moves forward per step (`window_segmenter_transform.py:219`). |
| `targeted_time_stamps_per_window` | int | Target number of time stamps per window. **Required** for model building — missing value aborts encoder/decoder construction (`encoder_decoder_utils.py:451`), and a `UserWarning` is raised if it differs from `windowed_data_specs.window_length` (`encoder_decoder_utils.py:463`). |

---

## `windowed_data_specs`

Parsed by `WindowShape` (`config_setup.py:173`).

| Field | Type | Role in code |
|---|---|---|
| `window_channels` | int | Channel count of the input window. In the vae_pipeline convention this is the channel dimension of the tensor fed to the encoder — the raw signal channels **plus** the validity-mask channels concatenated by `beta_VAE._prepare` (`models/vae_model.py:41-78`), i.e. typically 2× the raw signal channels (e.g. `30` for the 15-channel `flux_loop_flux` signal; the sibling `probe_diagnostics` config of the same VAE lists the raw `15`). Presence is required by `_chek_for_missing_attributes` (`encoder_decoder_utils.py:455`) and it is used as the tensor size in the test harness (`models/vae_model.py:312`). |
| `window_length` | int | Number of time samples per window. Used as the starting length in the conv output-size computation `_compute_conv_output_dim` (`encoder_decoder_utils.py:503`) and must equal `time_settings.targeted_time_stamps_per_window` (warning otherwise). |
| `permutation` | list(int) | Sequence of dimension indices specifying how to reorder the signal axes. Used in `src.common_transforms.general_transforms.ModelSpecificTransform`
---

## `training`

Parsed by `TrainingSettings` (`config_setup.py:187`).

| Field | Type | Role in code |
|---|---|---|
| `lr` | float | Learning rate of the `Adam` optimizer (`vae_pipeline.py:510`). |
| `num_epochs` | int | Number of epochs in the training loop (`vae_pipeline.py:86`) and `T_0` of the `CosineAnnealingWarmRestarts` scheduler (`vae_pipeline.py:515`). |
| `min_nr_epochs` | int | Minimum number of epochs before early stopping may trigger (`vae_pipeline.py:328`). |
| `patience` | int | Two uses: (1) the adaptive β-rescaling is only applied after `patience` epochs (`vae_pipeline.py:264`); (2) early stopping fires when validation loss has not improved for `patience` consecutive epochs and `epoch > min_nr_epochs` (`vae_pipeline.py:328`). |
| `min_increment` | float | Parsed into `SETTINGS.TRAINING.min_increment` but **not used** by the current training loop (reserved minimum-improvement threshold). |
| `dataloader_batch_size` | int | Batch size of the train and validation `DataLoader`s (`vae_pipeline.py:487-498`). |
| `num_workers` | int | `num_workers` of the train and validation `DataLoader`s (`vae_pipeline.py:490`). |
| `num_train_samples` | int | Number of shot IDs taken from the train split (`max_index_for_train` in `get_train_test_val_shots`, `vae_pipeline.py:411`). |
| `num_val_samples` | int | Number of shot IDs taken from the validation split (`max_index_for_val`, `vae_pipeline.py:412`). |
| `min_batch_size` | int | Present in the example JSON but **not parsed** by `TrainingSettings` — unused. |

---

## `beta-vae`

Parsed by `BetaVae` (`config_setup.py:106`).

| Field | Type | Role in code |
|---|---|---|
| `latent_dim` | int | Dimension `D` of the latent vector `z`; sets the `fc_mu` / `fc_logvar` head sizes (`models/vae_model.py:34-39`) and is required by model building (`encoder_decoder_utils.py:468`). This is the per-signal latent dimension the benchmark model sums over its input VAEs. |
| `beta` | float | Initial weight of the KL-divergence term in the masked loss `loss = recon + β·KL` (`vae_pipeline.py:63`, `models/vae_model.py:159`). During training β is **adaptively rescaled** every epoch to keep the KL/recon ratio near `0.1`, clipped to `[1e-5, 10]` (`vae_pipeline.py:256-277`); the history is saved in `loss_curves.json` and re-read by the visualization script (`vae_pipeline_visualization.py:157-158`). |
| `ref_freq` | float | Parsed into `SETTINGS.BETA_VAE.ref_freq` (`config_setup.py:113`) but **not consumed** by the current loss — reserved for frequency-domain scaling of the KL term. |

---

## `encoder`

Parsed by `Encoder` (`config_setup.py:124`). Selects the builder in
`EncoderDecoder.__init__` (`models/encoder_decoder.py:10-17`).

| Field | Type | Role in code |
|---|---|---|
| `type` | str | One of: `conv1d` → `build_conv1d_encoder_decoder` (auto-appends `flatten` + linear head and auto-mirrors a `ConvTranspose1d` decoder — only valid for encoders of pure conv1d + activation layers); `linear` → `build_linear_encoder_decoder` (decoder mirrored from the linear stack); `encoder_decoder` → `quick_build_from_config` (encoder and decoder built verbatim from the `layers` lists, `encoder_decoder_utils.py:119`). Missing/None `type` aborts model building (`encoder_decoder_utils.py:476`). |
| `activation_fn` | str | Activation name (e.g. `leaky_relu`) used for the auto-appended/mirrored layers and to validate the encoder composition when mirroring (`encoder_decoder_utils.py:282,320,400`). Required (`encoder_decoder_utils.py:472`). |
| `layers` | list[dict] | Layer specs of the encoder trunk, built by `SequentialBuilder` (`src/utils/layer_factory.py:80`). Format: `{"type": <registered layer>, "params": {<torch.nn kwargs>}}` (e.g. `conv1d`, `linear`, `flatten`, `unflatten`, `leaky_relu`, `gelu`, `conv_transpose1d`, ...). For `type: "encoder_decoder"` the last `linear` layer's `out_features` becomes `size_before_vae` (`encoder_decoder_utils.py:142-150`). |

## `decoder`

Parsed by `Decoder` (`config_setup.py:139`), optional at the
`Settings` level but **required when `encoder.type == "encoder_decoder"`**
(`KeyError: Decoder specs not available in config file`,
`encoder_decoder_utils.py:134`).

| Field | Type | Role in code |
|---|---|---|
| `layers` | list[dict] | Explicit decoder layer specs (`SequentialBuilder`). Must start from the latent space (first `linear` `in_features` = `beta-vae.latent_dim`) and end at the raw signal channels (last `conv_transpose1d` `out_channels` = raw channel count, e.g. 15 in `config_flux_loop_flux.json` — half of `window_channels`, since the mask channels are not reconstructed). When omitted with `encoder.type` of `conv1d`/`linear`, the decoder is auto-mirrored from the encoder (`encoder_decoder_utils.py:222,398`). |
| `type` | str | Parsed but not used in the current build path (mirrors `encoder.type` in the example configs). |
| `activation_fn` | str | Parsed but not used in the current build path. |

## `conv1d_encoder`

Parsed by `Conv1dEncoder` (`config_setup.py:257`), optional.

| Field | Type | Role in code |
|---|---|---|
| `conv1d_in_channels` | int | See below. |
| `conv1d_out_channels` | int | See below. |
| `kernel` | int | See below. |
| `stride` | int | See below. |
| `padding` | int | See below. |

If the section is present, **all five keys are required** — any missing
one aborts model building (`encoder_decoder_utils.py:481-495`). This is
a legacy single-conv1d spec; the current builders read `encoder.layers`
instead, so the section only acts as a validation marker today.

## `scheduler`

Parsed by `Scheduler` (`config_setup.py:276`), optional.

| Field | Type | Role in code |
|---|---|---|
| `mode` | str | Parsed but **not consumed** — the pipeline hard-codes `CosineAnnealingWarmRestarts` (`vae_pipeline.py:513-518`). Reserved for `ReduceLROnPlateau`-style specs (`mode`, `factor`, `threshold`, `threshold_mode`). |
| `factor` | float | See `mode`. |
| `threshold` | float | See `mode`. |
| `threshold_mode` | str | See `mode`. |

---

## `paths`

Parsed by `LocalPaths` (`config_setup.py:211`).

| Field | Type | Role in code |
|---|---|---|
| `data_split_csv_path` | str | TokaMark train/val/test shot split CSV, read by `get_train_test_val_shots` to build the shot ID lists (`vae_pipeline.py:414`). |
| `global_mean_std_path` | str | Directory containing `dict_signals_stats.yaml` — per-signal `mean`/`std` for the `StdScalingTransform` applied before windowing (`vae_pipeline.py:424-447`). |
| `data_output_directory` | str | Base results directory. The run directory is `data_output_directory + "conv1d_vae_" + <config file name without .json>` (`vae_pipeline.py:396`) and receives `best_vae_{signal}.pt`, `last_vae_{signal}.pt`, `loss_curves.json`, `model.json` and a copy of the config (`vae_pipeline.py:306-342,530-542`). These directories are what the benchmark config lists under `paths.vae_directory` (e.g. `src/vae_pipeline/data/New_VAEs/...`). |

---

## `local`

`true` → MAST diagnostics are loaded from the local zarr store
(`store_mast_settings={"base_local_zarr_path": ...}`) and passed as
`local_flag` to the MAST dataset (`vae_pipeline.py:452-457`).
`false` → data is fetched from the remote MAST. Default `true`.

## `cache_data`

Parsed into `SETTINGS.DATA.cache_data` (default `true`). The pipeline
calls `initialize_datasets` without forwarding it
(`vae_pipeline.py:453-459`, where the default `False` applies), so the
flag has **no effect** in the current code path; kept for
`CachedDataset` compatibility (`src/utils/utils.py:41-44`).

---

## Dimension consistency rules

For a config with `encoder.type == "encoder_decoder"` (the common case):

1. First `conv1d` `in_channels` of `encoder.layers` == `window_channels`
   (raw signal channels + mask channels, i.e. 2× raw).
2. `window_length` == `time_settings.targeted_time_stamps_per_window`.
3. The conv stack must compress `window_length` down to a small length;
   after `flatten`, the following `linear` `in_features` must equal
   (final conv out_channels × final conv length) — the product computed
   by `_compute_conv_output_dim` (`encoder_decoder_utils.py:499`).
4. `size_before_vae` = `out_features` of the last encoder `linear`
   (`encoder_decoder_utils.py:142-150`); it is the input size of
   `fc_mu` / `fc_logvar`.
5. First `linear` `in_features` of `decoder.layers` ==
   `beta-vae.latent_dim`.
6. Last `conv_transpose1d` `out_channels` of `decoder.layers` == raw
   signal channel count (half of `window_channels` in this convention).

For `conv1d` / `linear` encoder types, rules 4-6 are generated
automatically (mirrored decoder, appended heads) and the encoder may
only contain conv1d/linear + `activation_fn` layers
(`encoder_decoder_utils.py:282-289,400-407`).

---

## Config file conventions

- One config per signal: `config_<signal_name>.json` (e.g.
  `config_flux_loop_flux.json`). This checkout only contains
  `config_flux_loop_flux.json`; the per-signal configs used by the
  benchmark live under `src/vae_pipeline/data/New_VAEs/<dir>/config_*.json`
  (referenced by `src/benchmark/configs/*.json` via `paths.vae_directory`).
- Directory/name prefixes indicate the encoder family: `conv1d_vae_*`
  (convolutional encoder) and `linear_vae_*` (fully-connected encoder);
  the pipeline always prefixes the run directory with `conv1d_vae_`
  (`vae_pipeline.py:396`).
- Suffixes in names (e.g. `_25`, `_30`, `_55`) indicate the time window
  length in ms the VAE was trained on.
- Trained checkpoints (`best_vae_{signal_name}.pt`) sit next to the
  config JSON in the run directory; that is the layout the benchmark
  pipeline expects when loading VAEs
  (`src/benchmark/utils_new.py:73-77`).
