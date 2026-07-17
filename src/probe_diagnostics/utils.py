#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path




def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_mu_vs_shot_id(latent_data, shot_min, shot_max, save_path):
    """
    Plot latent coordinates versus shot_id
    for all shots in [shot_min, shot_max].
    """

    shot_ids = sorted(
        shot_id
        for shot_id in latent_data
        if shot_min <= shot_id <= shot_max
    )

    if len(shot_ids) == 0:
        raise ValueError(
            f"No shots found in interval [{shot_min}, {shot_max}]"
        )

    first_shot = shot_ids[0]
    window_indices = sorted(latent_data[first_shot].keys())

    mu_matrix = np.stack([
        latent_data[shot_id][window_indices[0]]["mu"]
        for shot_id in shot_ids
    ])
    
    latent_dim = mu_matrix.shape[1]

    mu_mean = mu_matrix.mean(axis=0)
    mu_std = mu_matrix.std(axis=0)

    fig, axes = plt.subplots(
        latent_dim,
        1,
        figsize=(12, 2.5 * latent_dim),
        sharex=True,
    )

    if latent_dim == 1:
        axes = [axes]

    # tick_idx = np.arange(0, len(shot_ids), 10)

    for d in range(latent_dim):

        axes[d].plot(
            shot_ids,
            mu_matrix[:, d],
            marker="o",
            linewidth=1,
        )

        axes[d].axhline(
            mu_mean[d] + 3 * mu_std[d],
            color="red",
            linestyle="--",
            linewidth=1,
            label="+3 std",
        )

        axes[d].axhline(
            mu_mean[d] - 3 * mu_std[d],
            color="red",
            linestyle="--",
            linewidth=1,
            label="-3 std",
        )

        axes[d].set_ylabel(f"$z_{d}$")
        axes[d].grid(True)

        axes[d].set_xticks(shot_ids)
        axes[d].set_xticklabels(
            shot_ids,
            rotation=90,
        )

        axes[d].tick_params(
            axis="x",
            labelbottom=True,
            labelsize=6,
        )

    fig.suptitle(
        f"Latent coordinates ({shot_min}-{shot_max})",
        y=0.995,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_residual_vs_shot_id(
    latent_data,
    shot_min,
    shot_max,
    save_path,
):
    """
    Plot residual versus shot_id.
    """

    shot_ids = sorted(
        shot_id
        for shot_id in latent_data
        if shot_min <= shot_id <= shot_max
    )

    if len(shot_ids) == 0:
        raise ValueError(
            f"No shots found in interval [{shot_min}, {shot_max}]"
        )

    residuals = []

    for shot_id in shot_ids:
        shot_residuals = [latent_data[shot_id][0]["residual"].item()]
        residuals.append(shot_residuals)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        shot_ids,
        residuals,
        marker="o",
        linewidth=1,
    )

    ax.set_xlabel("shot_id")
    ax.set_ylabel("residual")
    ax.grid(True)

    # tick_idx = np.arange(0, len(shot_ids), 1)

    ax.set_xticks(shot_ids)
    ax.set_xticklabels(
        shot_ids,
        rotation=90,
        fontsize=10,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def get_shot_in_campaign(campaign):
    summary  = pd.read_parquet('https://mastapp.site/parquet/level2/shots')
    return summary.loc[summary["campaign"] == campaign, "shot_id"]

def get_shots_for(key, value):
    summary  = pd.read_parquet('https://mastapp.site/parquet/level2/shots')
    return summary.loc[summary[key] == value, "shot_id"]

def plot_zoom_shot_interval(shot_min, shot_max, pickle_file):

    latent_data = load_pickle(pickle_file)

    parent_dir = os.path.dirname(pickle_file)
    
    plot_mu_vs_shot_id(
        latent_data,
        shot_min,
        shot_max,
        os.path.join(parent_dir, "mu_vs_shot_id.pdf"),
    )

    plot_residual_vs_shot_id(
        latent_data,
        shot_min,
        shot_max,
         os.path.join(parent_dir, "residual_vs_shot_id.pdf"),
    )


def plot_merged_latent_spaces(
    pickle_files,
    window_index,
    colors,
    labels
):
    save_path = None
    
    if len(colors)!=len(pickle_files)!=len(labels):
        raise ValueError(f"Not enough colors for all the latent spaces")
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    for i, pickle_file in enumerate(pickle_files):
        
        latent_data = load_pickle(pickle_file)

        shot_ids = sorted(
            shot_id
            for shot_id in latent_data
            if window_index in latent_data[shot_id]
        )

        if len(shot_ids) == 0:
            print(
                f"No shots found for window "
                f"{window_index} in {pickle_file}"
            )
            continue

        mu_matrix = np.stack([
            latent_data[shot_id][window_index]["mu"]
            for shot_id in shot_ids
        ])

        if mu_matrix.ndim != 2:
            raise ValueError(
                f"Wrong dimensionality of mu_matrix: "
                f"{mu_matrix.ndim}"
            )

        latent_dim = mu_matrix.shape[1]

        if latent_dim != 3:
            raise ValueError(
                f"Expected latent_dim=3, "
                f"got {latent_dim}"
            )

        z_test = (mu_matrix - mu_matrix.mean(axis=0))/mu_matrix.std(axis=0)
        idx = (abs(z_test)<3).all(axis=1)
        x = mu_matrix[idx, 0]
        y = mu_matrix[idx, 1]
        z = mu_matrix[idx, 2]

        label = Path(pickle_file).stem

        ax.plot(
            x,
            y,
            z,
            marker="o",
            markersize=2,
            color = colors[i],
            linewidth=0,
            label=labels[i],
        )
        
        if save_path is None:
            parent_dir = os.path.dirname(pickle_file)
            save_path = os.path.dirname(parent_dir)

    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_zlabel(r"$z_3$")

    ax.legend()

    if parent_dir is None:
        parent_dir = os.path.dirname(pickle_file)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shot_min",
        type=int,
        default=None,
        help="Minimum shot_id",
    )

    parser.add_argument(
        "--shot_max",
        type=int,
        default=None,
        help="Maximum shot_id",
    )

    parser.add_argument(
        "--pickle_file",
        nargs="+",
        default=None,
        help="Path to latent_data pickle",
    )

    args = parser.parse_args()
    plot_merged_latent_spaces(args.pickle_file, 10, ["blue", "red"], ["SS_Beam", "SW_Beam"])
        
        