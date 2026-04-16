# VAE Fairmast

Directory structure:
src/vae_pipeline
src/benchmark

Data repository:
https://huggingface.co/datasets/UKAEA-IBM-STFC/tokamark-dataset

## Getting started
git clone --recurse-submodules git@gitlab.stfc.ac.uk:hncdi-fusion-plasma-modelling/VAE_fairmast.git



# src directory content

## vae_pipeline
PyTorch pipeline for training Variational Auto Encoder (VAE) architectures for learning the latent space representation of signals from the Mega Ampere Spherical Tokamak (MAST) experiments.

### Encoder
The encoder compresses input data **x** into the latent space **z**, where dim(**z**)< dim(**x**).

The encoder architectures used in the training are:
- A stack of conv1d layers.
- A stack of conv2d layers.
- A series of dense layers.


### Decoder
The decoder decompresses **z** to return **x**.

The decoder applies the inverse encoder transform in reverse order.


## How to use it
```python src/vae_pipeline/vae_pipeline.py --config_file_path src/vae_pipeline/configs/config*.json```

Use a `config*.json` file in `src/vae_pipeline/config` to configure your training session.

In the configuration file, adjust the timing structure to define the time windows used for training. You must set both **`x_window_sec`** and **`y_window_sec`** for the input and target data, respectively.

In a VAE, the input and target coincide. However, due to the way the code is structured, **`y_window_sec`** must be set to the sampling period of the signal.

The **`x_window_sec`** window is slid over the signal trace using the step size defined by **`stride_sec`**.

It is important to explicitly set the number of time stamps per window using the key **`targeted_time_stamps_per_window`**. This value is given by the ratio between the size of **`x_window_sec`** and the sampling period, rounded down to the nearest integer.

The key **`windowed_data_specs`** also defines the structure of each window, namely:
- **`window_channels`**: number of channels per window  
- **`window_length`**: length of the window  

In most cases, **`window_length`** coincides with **`targeted_time_stamps_per_window`**. However, for 3D signals, each time stamp corresponds to a 2D image of shape `[window_channels, window_length]`.

Fill in all remaining key–value pairs in the configuration file as required.

### Decoder configuration (optional)

If you want to explicitly specify a decoder architecture, you can do so by adding a decoder definition that mirrors the encoder structure (possibly with different parameter choices), for example:

```json
"decoder": {
  "type": "your_type",
  "activation_fn": "your_function",
  "layers": [
    { "type": "...", "params": { ... } },
    { "type": "..." }
  ]
}
```
## benchmark
Code to evaluate trained VAEs over tasks defined in fairmast_data_process.src.benchmark.

For more details on the benchmark study see arXiv:2602.10132 

### How to use it
```python src/benchmark/benchmark_pipeline.py --config_benchmark_file_path path_to_json_benchmark_file --config_task_file_path fairmast_data_processing/src/MAST_benchmark/tasks_configs/.yaml```






