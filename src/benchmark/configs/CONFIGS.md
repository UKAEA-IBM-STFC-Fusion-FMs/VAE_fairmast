# Benchmark Configuration Reference

Field-by-field documentation of the benchmark JSON configs (`src/benchmark/configs/task*_config*.json`), based on how each field is
consumed in the code. The config is parsed by `SettingsBenchmark` (`src/benchmark/configs/benchmark_setup.py:56`) and passed around as `SETTINGS` to the pipeline, evaluator and  isualization
scripts.

## Top-level fields

| Field | Type | Required | Parsed as | Purpose |
|---|---|---|---|---|
| `paths` | object | yes (KeyError if absent) | `SETTINGS.LOCAL_PATHS` | File/directory locations: data splits, stats, VAE models, output. See [paths](#paths). |
| `training` | object | yes (KeyError if absent) | `SETTINGS.TRAINING` | Hyper-parameters for the training loop. See [training](#training). |
| `signal_model`, `mask_model`, `end_model` | objects | yes | `SETTINGS.MODEL.*` | Layer specs of the three-branch `BenchmarkModel`. See [model sections](#model-sections). |
| `local` | bool | no (default `True`) | `SETTINGS.local` | Whether MAST data is read from a local zarr store. See below. |
| `output_signals_len` | list[int] | no (default `None`) | `SETTINGS.output_signals_len` | Per-output-signal lengths used to split the concatenated target tensor. See below. |
| `cache_data` | bool | no (default `False`) | `SETTINGS.cache` | Dataset caching flag. See below. |
| `task` | int | yes (KeyError if absent) | `SETTINGS.task` | TokaMark task identifier: `1` = reconstruction, `2` = dynamics, `3` = profiles & dynamics. |
| `fine_tuning` | bool | no | *(not read)* | Marker used only in the `*_config_with_fine_tuning.json` variants, indicating the benchmark model is trained on top of VAEs that were fine-tuned for the target task. Not consumed by `SettingsBenchmark`. |

\* `SettingsBenchmark` requires the trio `signal_model` + `mask_model` + `end_model`,
(`benchmark_setup.py:71`) or `KeyError` is raised.

### `local`

`true` → MAST diagnostics are loaded from the local zarr store
(`store_mast_settings={"base_local_zarr_path": ...}`) and passed as
`local_flag` to the MAST dataset (`benchmark_pipeline.py:382-387`).
`false` → data is fetched from the remote MAST. Default `true`.

### `output_signals_len`

One integer per output signal of the task, in the same order as the `output_name` list of the task YAML. Used by`decode_reco_signals` (`benchmark_visualization_new.py:314`) to slice the
flattened concatenated target tensor back into individual signals:

- for an output signal **with** a VAE, the entry is the VAE **latent dimension**;
- for an output signal **without** a VAE, the entry is the signal length in
  **real space after flattening**.

The sum of the entries must equal the final `out_features` of
`end_model` (e.g. `task2_1`: `[35, 6, 5, ...]` sums to 131).

### `cache_data`

Parsed into `SETTINGS.cache`. Note: the current pipeline hard-codes `cache_data=False` in its `initialize_datasets` call (`benchmark_pipeline.py:388`), so this flag has **no effect** in the
current code paths; it is kept for compatibility with `CachedDataset` wrapping (`src/utils/utils.py:41-44`).

### `task`

Identifies which TokaMark task group the config belongs to (`1`, `2` or `3`). Stored on `SETTINGS.task`; the actual task definition is selected at runtime via `--config_task_file_path` (the TokaMark YAML), so `task` is currently informational.

---

## `training`

Parsed by `TrainingSettings` (`benchmark_setup.py:101`). Missing keys print a warning and are set to `None`.

| Field | Type | Role in code |
|---|---|---|
| `lr` | float | Learning rate of the `Adam` optimizer (`benchmark_pipeline.py:457`). Also sets the scheduler floor `eta_min = int(lr/50)` (`benchmark_pipeline.py:463`). |
| `num_epochs` | int | Number of epochs in the training loop (`benchmark_pipeline.py:122`) and `T_0` of the `CosineAnnealingWarmRestarts` scheduler (`benchmark_pipeline.py:461`). |
| `min_nr_epochs` | int | Minimum number of epochs before early stopping may trigger (`benchmark_pipeline.py:280`). |
| `patience` | int | Early stopping fires when validation loss has not improved for `patience` consecutive epochs **and** `epoch > min_nr_epochs` (`benchmark_pipeline.py:280`). The best model state is saved to `output_directory/best_model.pt`. |
| `dataloader_batch_size` | int | Batch size of the train and validation `DataLoader`s (`benchmark_pipeline.py:418-429`). |
| `num_workers` | int | `num_workers` of the train and validation `DataLoader`s (`benchmark_pipeline.py:421`). |
| `num_train_samples` | int | Number of shot IDs taken from the train split (`max_index_for_train` in `get_train_test_val_shots`, `benchmark_pipeline.py:335`). |
| `num_val_samples` | int | Number of shot IDs taken from the validation split (`max_index_for_val`, `benchmark_pipeline.py:336`). |

---

## `paths`

Parsed by `LocalPaths` (`benchmark_setup.py:136`). Missing keys print a warning and are set to `None`.

| Field | Type | Role in code |
|---|---|---|
| `data_split_csv_path` | str | TokaMark train/val/test shot split CSV, read by `get_train_test_val_shots` to build the shot ID lists (`benchmark_pipeline.py:338`, `src/utils/utils.py:117`). |
| `global_mean_std_path` | str | Directory containing `dict_signals_stats.yaml` — per-signal `mean`/`std` used to build the `StdScalingTransform` for every input/actuator/output signal (`benchmark_pipeline.py:359-378`). |
| `vae_directory` | str | Base directory for the pretrained VAEs. Each entry of the `*_vae_models` lists is joined to it to form the path of a VAE `config.json` (`utils_new.py:122`). The matching weights file `best_vae_<signal_name>.pt` is loaded from the same sub-directory (`utils_new.py:73-74`). |
| `output_directory` | str | Base results directory. A sub-folder named after the config file (without `.json`) is appended at runtime, and receives `best_model.pt`, `loss_curves.json`, `model.json`, a copy of the config, and the evaluation/visualization figures (`benchmark_pipeline.py:328`, `benchmark_evaluator.py:90`, `benchmark_visualization_new.py:514`). |
| `input_vae_models` | list[str] | Sub-paths (relative to `vae_directory`) of the `config.json` files of the VAEs encoding the **input** signals. A 1-to-1 correspondence with the task's `input_name` signals is **enforced** — a `ValueError` is raised if a signal has no matching VAE (`utils_new.py:128-133`). Input signals without a VAE are not allowed. |
| `actuator_vae_models` | list[str] | Same as above for **actuator** signals (`actuator_name`). Empty list → no actuators, no actuator VAEs required. If actuators are configured, all of them must have a VAE (enforced). |
| `output_vae_models` | list[str] | Same for **output** signals (`output_name`). Empty list → outputs stay in **real space** (flattened to 2D, `utils_new.py:291`); if listed, each output signal must have a VAE and is encoded to its latent space. A mismatch for outputs does not raise (unlike inputs/actuators). |

Matching rule: a signal is paired with the VAE whose sub-path **contains
the signal name** (`signal_name in model_path`, `utils_new.py:120`).

---

## Model sections

Layer specs are consumed by `SequentialBuilder`
(`src/utils/layer_factory.py:80`), which builds an `nn.Sequential` from
the list. Each entry is:

```json
{"type": "<layer name>", "params": {<kwargs of the torch.nn layer>}}
```

`params` is optional for argument-less layers (e.g. `{"type": "gelu"}`).
Registered names include `linear`, `conv1d`, `conv2d`, `relu`,
`leaky_relu`, `tanh`, `sigmoid`, `gelu`, `layer_norm`, `batch_norm1d`,
`dropout`, `max_pool1d`, `adaptive_avg_pool1d`, `flatten` (see
`LayerFactory._register_default_layers`, `layer_factory.py:14`).

### `signal_model` → `SETTINGS.MODEL.signal_layers`

Built as `signal_mlp` in `BenchmarkModel` (`benchmark_model_new.py:37`).
Encodes the **concatenated latent representations** of all input and
actuator VAEs.

- First `linear` layer's `in_features` **must equal** the sum of the
  `latent_dim` of all input + actuator VAEs; checked at training start
  (`benchmark_pipeline.py:109-120`, raises `ValueError` on mismatch).
- Last `linear` layer's `out_features` is the signal feature width `F`.

### `mask_model` → `SETTINGS.MODEL.mask_layers`

Built as `mask_mlp` (`benchmark_model_new.py:39`). Encodes the
**concatenated validity masks** of the same signals.

- `in_features` = number of mask entries after concatenation: one per
  input/actuator VAE (sample-level mask) or one per element for signals
  without a VAE (element-wise mask).
- `out_features` must be either `F` (gamma only, applied as
  `h = h_mask * h_signal`) or `2*F` (gamma + beta, applied as
  `h = (1 + gamma) * h_signal + beta`); validated at init
  (`benchmark_model_new.py:80-82`) and in `forward`
  (`benchmark_model_new.py:89-99`).

### `end_model` → `SETTINGS.MODEL.end_layers`

Built as `fusion_mlp` (`benchmark_model_new.py:41`). The prediction head
that maps the modulated signal features to the **concatenated target
representation** (latent for VAE-compressed outputs, real for the rest).

- `in_features` = `F` (signal feature width).
- `out_features` = `sum(output_signals_len)`.

### `model` → `SETTINGS.MODEL.model_layers`

Layer spec of a plain single-MLP benchmark model. Used as a fallback when
building `BenchmarkModel` fails (`benchmark_pipeline.py:449-454`) and in
the "old" code path (`benchmark_visualization_old.py:607`). A config with
`model` does not need `signal_model`/`mask_model`/`end_model`.

---

## Dimension consistency rules (across fields)

For the three-branch model the following must hold (enforced or expected):

1. `signal_model` first `in_features` == Σ `latent_dim` of all VAEs in
   `input_vae_models` + `actuator_vae_models`.
2. `mask_model` `out_features` ∈ {`signal_model` last `out_features`,
   2 × `signal_model` last `out_features`}.
3. `end_model` first `in_features` == `signal_model` last `out_features`.
4. `end_model` last `out_features` == `sum(output_signals_len)`.
5. `output_signals_len` has one entry per task output signal; each entry
   is the VAE `latent_dim` (if the signal is in `output_vae_models`) or
   the flattened real-space length (otherwise).

---

## Config file conventions

- `task1_1_config.json` … `task3_1_config.json` — one config per
  benchmark experiment; run with
  `python src/benchmark/benchmark_pipeline.py --config_benchmark_file_path <json> --config_task_file_path <task yaml>`.
- `task1_3_config_gamma_factor.json` — variant where the mask branch
  outputs gamma + beta (`mask_model` width = 2 × `signal_model` width).
- `task*_config_with_fine_tuning.json` — variants trained on top of
  task-fine-tuned VAEs; carry the (currently unused) `fine_tuning: true`
  flag.
