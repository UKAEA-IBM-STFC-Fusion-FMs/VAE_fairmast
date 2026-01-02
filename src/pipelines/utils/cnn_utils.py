import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v3 as iio
import matplotlib.gridspec as gridspec
from torch.utils.data._utils.collate import default_collate

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipelines.transforms.shot_level_transforms.cnn_transform import (
    CNNTransform,
)

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
        "..",
        "..",
    )
)  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# print(f"REPO_ROOT: {REPO_ROOT}")


# ----------------------------------------------------------------------------------------------------------------------
# Determine device to train on

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def flatten_then_collate(batch):
    print(f"Collating batch of size {len(batch)}")

    # Flatten the batch of lists into a single list
    flattened_batch = []
    if isinstance(batch[0], list):
        flattened_batch = [item for sublist in batch for item in sublist]
        print(
            f"Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}"
        )
        # for bat in flattened_batch:
        #     print(
        #         f"Shapes in batch = {[arr.shape for arr in bat[0]]} and {[arr.shape for arr in bat[1]]}"
        #     )

    # Use the default collate function
    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ----------------------------------------------------------------------------------------------------------------------
# INVERSE STDSCALING
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def flatten_blocks(list_y):
    """
    Flattens each block of predictions into individual series.

    """
    new_list_y = []
    for block in list_y:
        if isinstance(block, torch.Tensor):
            block = block.detach().cpu().numpy()  # detach, move to CPU, convert to numpy
        else:
            block = np.asarray(block)
            
        if block.ndim == 1:
            new_list_y.append(block)
        else:
            new_list_y += [
                np.squeeze(s, axis=1) for s in np.split(block, block.shape[1], axis=1)
            ]
    return new_list_y


# ----------------------------------------------------------------------------------------------------------------------
def get_cnn_order_scaling(
    LOCAL_FLAG,
    test_shots_,
    source_signal_list,
    signal_transform_map,
    shot_transform_without_CNN,
    verbose = True
):
    transform_temp = CNNTransform()
    dataset_temp = MastDataset(
        local=LOCAL_FLAG,
        shots_list=test_shots_,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform=shot_transform_without_CNN,
    )

    for l in range(len(dataset_temp)):
        data_temp = dataset_temp[l]
        data_temp = transform_temp(data_temp)
        try:
            order_var_for_inv_std = [
                item for arr in flatten_blocks(transform_temp.var_groups["y"]) for item in arr
            ]
            return order_var_for_inv_std
        except Exception as e:
            continue
            # print(f"Skipping {l} because shot not trainable: {e}")



# ----------------------------------------------------------------------------------------------------------------------
def inverse_standardize(flat_data, order_vars, dict_mean, dict_std):
    """
    Inverse standardize a list of flattened arrays using provided means and stds.

    Args:
        flat_data (list of np.ndarray): Flattened predicted/true arrays.
        order_vars (list): Variable names, aligned with flat_data.
        dict_mean (dict): Mapping var -> mean.
        dict_std (dict): Mapping var -> std.

    Returns:
        list of np.ndarray: Inverse standardized arrays.
    """
    new_flat = []
    for var, data in zip(order_vars, flat_data):
        mean = dict_mean[var]
        std = dict_std[var]

        # prevent division by zero (replace 0 with 1)
        std_safe = np.where(std == 0, 1.0, std)

        # inverse transform
        new_data = data.T * std_safe[..., None] + mean[..., None]
        new_flat.append(np.squeeze(new_data.T))

    return new_flat


# ----------------------------------------------------------------------------------------------------------------------
# VISUALISATION
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def plot_shot_gif(
    flat_preds,
    flat_trues,
    order_var_list,
    avg_test_loss,
    shot_idx,
    ref_freq,
    out_dir="shot_gifs",
    fps=8,
    cleanup=True,
):
    """
    Create an animated GIF showing predictions vs ground truth over time for a given shot.

    Parameters
    ----------
    list_y_pred, list_y_true : list of arrays
        Each array shape (T, N) or (T,)
    shot_idx : int or str
        Identifier for the shot (used in filenames)
    ref_freq : float
        Time step in ms
    out_dir : str
        Directory to store frames and GIF
    fps : int
        Frames per second of the animation
    cleanup : bool
        If True, delete individual frame PNGs after creating the GIF
    """
    out_folder = Path(out_dir) / f"shot_gifs"
    frame_dir = out_folder / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    n_outputs = len(flat_preds)
    T = min(y.shape[0] for y in flat_preds)  # number of time steps (shortest sequence)

    # Determine global min/max for 2D profiles
    # profile_min, profile_max = None, None
    max_list = [ max(v_pred, v_true) for v_pred, v_true in zip([arr.max() for arr in flat_preds], [arr.max() for arr in flat_trues]) ]
    min_list = [ min(v_pred, v_true) for v_pred, v_true in zip([arr.min() for arr in flat_preds], [arr.min() for arr in flat_trues]) ]
    # for y_pred, y_true in zip(flat_preds, flat_trues):
    #     if y_pred.ndim == 2:
    #         vmin = min(y_true.min(), y_pred.min())
    #         vmax = max(y_true.max(), y_pred.max())
    #         if profile_min is None or vmin < profile_min:
    #             profile_min = vmin
    #         if profile_max is None or vmax > profile_max:
    #             profile_max = vmax

    # Row height ratios for subplot layout
    row_heights = [0.5 if y.ndim == 1 else 1.0 for y in flat_preds]
    fig_height = sum(row_heights) * 4

    frame_paths = []

    # Loop over time steps
    for t in range(1, T + 1):
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(
            n_outputs, 2, width_ratios=[1, 1], height_ratios=row_heights, hspace=0.4
        )

        viz_max_t = T * ref_freq
        time_ms = np.arange(t) * ref_freq

        for j, (var, y_pred, y_true, mse, var_min, var_max) in enumerate(
            zip(order_var_list, flat_preds, flat_trues, avg_test_loss, min_list, max_list)
        ):
            viz_min_y = min(y_true.min(), y_pred.min()) - abs(
                min(y_true.min(), y_pred.min()) * 0.05
            )
            viz_max_y = max(y_true.max(), y_pred.max()) + abs(
                max(y_true.max(), y_pred.max()) * 0.05
            )

            yp = y_pred[:t]
            yt = y_true[:t]

            ax_gt = fig.add_subplot(gs[j, 0])
            ax_pred = fig.add_subplot(gs[j, 1])

            if yp.ndim == 1:
                # 1D time series
                ax_gt.plot(time_ms, yt, lw=2, color="blue")
                ax_gt.set_title(f"{var} - Ground Truth")
                ax_gt.set_xlim(0, viz_max_t)
                ax_gt.set_ylim(viz_min_y, viz_max_y)
                # ax_gt.set_xlabel("Time (ms)")
                # ax_gt.set_ylabel("Value")

                ax_pred.plot(time_ms, yp, lw=2, color="orange")
                ax_pred.set_title(f"{var} - Prediction MSE {round(mse, 3)}")
                ax_pred.set_xlim(0, viz_max_t)
                ax_pred.set_ylim(viz_min_y, viz_max_y)
                # ax_pred.set_xlabel("Time (ms)")
                # ax_pred.set_ylabel("Value")

            elif yp.ndim == 2:
                # 2D profile
                D = yp.shape[1]

                ax_gt.imshow(
                    yt.T,
                    aspect="auto",
                    origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_gt.set_title(f"{var} - Ground Truth")
                ax_gt.set_xlim(0, viz_max_t)
                # ax_gt.set_xlabel("Time (ms)")
                # ax_gt.set_ylabel("Profile index")

                ax_pred.imshow(
                    yp.T,
                    aspect="auto",
                    origin="lower",
                    extent=[time_ms[0], time_ms[-1], 0, D],
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_pred.set_title(f"{var} - Prediction MSE {round(mse, 3)}")
                ax_pred.set_xlim(0, viz_max_t)
                # ax_pred.set_xlabel("Time (ms)")
                # ax_pred.set_ylabel("Profile index")

            elif yp.ndim == 3:
                # 3D image

                img = yt[-1]
                # print(img)
                ax_gt.imshow(
                    img,
                    aspect="auto",
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_gt.set_title(f"{var} - Ground Truth")

                img = yp[-1]
                # print(img)
                ax_pred.imshow(
                    img,
                    aspect="auto",
                    cmap="viridis",
                    vmin=var_min,
                    vmax=var_max,
                )
                ax_pred.set_title(f"{var} - Prediction MSE {round(mse, 3)}")

            else:
                raise ValueError(
                    f"Unsupported shape {yp.shape} (expected (T,) or (T,D))"
                )

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        frame_path = frame_dir / f"frame_{t:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)

    # Build GIF
    gif_path = out_folder / f"shot_{shot_idx}.gif"
    # Duration per frame in secondst    frames = [iio.imread(frame) for frame in frame_paths]
    frames = [iio.imread(frame) for frame in frame_paths] + [
        iio.imread(frame_paths[-1])
    ]
    durations = [1 / fps] * (len(frames) - 2) + [5] + [1 / fps]  # last frame = 5 sec
    iio.imwrite(
        gif_path,
        frames,
        format="GIF",
        duration=durations,
        # loop=0
    )

    # durations = [1 / fps] * (len(frame_paths) - 1) + [5]  # last frame = 5 sec
    # with iio.get_writer(gif_path, mode="I", loop=0) as writer:
    #     for frame, dur in zip(frame_paths, durations):
    #         writer.append_data(iio.imread(frame), duration=dur)

    # Optional cleanup
    if cleanup:
        for frame in frame_paths:
            frame.unlink()

    print(f"Saved GIF: {gif_path}")
