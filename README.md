# VAE Fairmast

Repository for a PyTorch pipeline that trains on the MAST dataset to learn its latent space representation.

The model under training is a Variational Auto Encoder: encoder-> latent space -> decoder.

## Encoder
The encoder compresses input data **x** into the latent space **z**, where dim(**z**)< dim(**x**).

The encoer architectures used in the training were 2:
- The first and most used one is a stack of conv1d layers.
- The second and less frequent one is a series of dense layers.

## Decoder
The decoder decompresses **z** to return **x**.

The decoder applies the inverse encoder transform in reverse order.

## Getting started
git clone --recurse-submodules git@gitlab.stfc.ac.uk:hncdi-fusion-plasma-modelling/VAE_fairmast.git

## Start a training session
Use config_template.json to set your training session. 

In the config file, adjust the timing structure to set the time windows for your trining. You must set both "x_window_sec" and "y_window_sec" for your inout and target data, respectivelly.
In a VAE, the input and target concide. For the way the code is structured, set the "y_window_sec" to the sample frequency of your signal. 
The  x_window_sec is slid over the signal trace with a "stride_sec" pass. 
It is important to set the number of time stamps per window by using the key "targeted_time_stamps_per_window". This is give by the ratio between
the size of "x_window_sec" and the sample frequency, round it down to the closest integer value. 

Fill in all other keys:values pairs. 

If you want to pass a decoder architecture, this can be done by adding a decoder structure that resambles the encoder structure (but with different choices of the params):
"decoder":{
    "type": "your_type",
    "activation_fn": "your_function",
    "layers": [
      {"type": " ... ", "params": { ... }}, 
      {"type": "..."}, 
      ...
    ]
  },

  If a decoder structure is not passed, then the code tries to write one automatically from the encoder. This works only for certain encoder architectures.

  From the home of your project: ```python src/pipelines/vae_pipeline.py --config_file_path src/pipelines/config/config_template.json```



