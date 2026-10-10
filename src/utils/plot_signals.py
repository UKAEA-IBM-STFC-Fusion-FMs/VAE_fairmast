import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_psi(
    reco: torch.Tensor,
    target: torch.Tensor,
    save_path: str = "reco_vs_target.pdf",
    n_contours: int = 15,
) -> None:
    """
    Plot target, reconstruction, and reconstruction error for EFIT psi.

    For C == 1, the figure contains three columns:
        target | reconstruction | difference

    For C > 1, the figure contains three rows:
        first row  : target
        second row : reconstruction
        third row  : difference

    The difference is defined as:

        difference = reconstruction - target

    Parameters
    ----------
    reco : torch.Tensor
        Reconstructed signal with shape [1, C, 65, 65].
    target : torch.Tensor
        Target signal with shape [1, C, 65, 65].
    save_path : str, optional
        Path where the figure is saved.
    n_contours : int, optional
        Number of contour levels.
    """
    expected_spatial_shape = (65, 65)

    # -------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------
    if not isinstance(reco, torch.Tensor):
        raise TypeError(
            f"reco must be a torch.Tensor, got {type(reco).__name__}."
        )

    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"target must be a torch.Tensor, got {type(target).__name__}."
        )

    if reco.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "Expected reco and target to have shape [1, C, 65, 65], "
            f"but got {tuple(reco.shape)} and {tuple(target.shape)}."
        )

    if reco.shape != target.shape:
        raise ValueError(
            "Reco and target must have the same shape, "
            f"but got {tuple(reco.shape)} and {tuple(target.shape)}."
        )

    if reco.shape[0] != 1:
        raise ValueError(
            "The function expects a batch size of 1, "
            f"but got {reco.shape[0]}."
        )

    if tuple(reco.shape[-2:]) != expected_spatial_shape:
        raise ValueError(
            f"Expected spatial shape {expected_spatial_shape}, "
            f"but got {tuple(reco.shape[-2:])}."
        )

    if reco.shape[1] < 1:
        raise ValueError("The channel dimension must be at least 1.")

    if n_contours < 2:
        raise ValueError(
            f"n_contours must be at least 2, got {n_contours}."
        )


    # Remove only the batch dimension [C, 65, 65]
    reco_array = reco.detach().cpu().numpy()[0]
    target_array = target.detach().cpu().numpy()[0]

    if not np.all(np.isfinite(reco_array)):
        raise ValueError("reco contains NaN or infinite values.")

    if not np.all(np.isfinite(target_array)):
        raise ValueError("target contains NaN or infinite values.")

    difference_array = reco_array - target_array
    n_channels = reco_array.shape[0]

    # -------------------------------------------------------------
    # Target and reconstruction colour scale
    # -------------------------------------------------------------
    psi_vmin = min(
        float(target_array.min()),
        float(reco_array.min()),
    )
    psi_vmax = max(
        float(target_array.max()),
        float(reco_array.max()),
    )

    if psi_vmax > psi_vmin:
        psi_contour_levels = np.linspace(
            psi_vmin,
            psi_vmax,
            n_contours,
        )
    else:
        # Contouring is not meaningful for a constant field.
        psi_contour_levels = None

    # -------------------------------------------------------------
    # Difference colour scale
    # -------------------------------------------------------------
    difference_limit = float(
        np.max(np.abs(difference_array))
    )

    if difference_limit == 0.0:
        difference_limit = np.finfo(float).eps

    difference_vmin = -difference_limit
    difference_vmax = difference_limit

    difference_contour_levels = np.linspace(
        difference_vmin,
        difference_vmax,
        n_contours,
    )

    # -------------------------------------------------------------
    # Coordinates
    # -------------------------------------------------------------
    z, r = np.mgrid[
        0 : target_array.shape[-1],
        0 : target_array.shape[-2],
    ]

    # =============================================================
    # Single-channel case
    # =============================================================
    if n_channels == 1:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 5),
            constrained_layout=True,
        )

        target_axis = axes[0]
        reco_axis = axes[1]
        difference_axis = axes[2]

  
        psi_image = target_axis.imshow(
            target_array[0],
            origin="lower",
            cmap="viridis",
            vmin=psi_vmin,
            vmax=psi_vmax,
        )

        if psi_contour_levels is not None:
            target_axis.contour(
                r,
                z,
                target_array[0],
                levels=psi_contour_levels,
                colors="white",
                linewidths=0.5,
            )

        target_axis.set_title(
            r"Input EFIT-$\psi(R,Z)$"
        )

       
        reco_axis.imshow(
            reco_array[0],
            origin="lower",
            cmap="viridis",
            vmin=psi_vmin,
            vmax=psi_vmax,
        )

        if psi_contour_levels is not None:
            reco_axis.contour(
                r,
                z,
                reco_array[0],
                levels=psi_contour_levels,
                colors="white",
                linewidths=0.5,
            )

        reco_axis.set_title(
            r"Reconstructed EFIT-$\psi(R,Z)$"
        )

       
        difference_image = difference_axis.imshow(
            difference_array[0],
            origin="lower",
            cmap="RdBu_r",
            vmin=difference_vmin,
            vmax=difference_vmax,
        )

        difference_axis.contour(
            r,
            z,
            difference_array[0],
            levels=difference_contour_levels,
            colors="black",
            linewidths=0.5,
        )

        difference_axis.set_title(
            r"Difference: reconstructed $-$ target"
        )

        # Axis formatting
        for axis in axes:
            axis.set_xlabel(r"$R$ ")
            axis.set_ylabel(r"$Z$ ")
            axis.set_aspect("equal")

        # Shared colorbar for target and reconstruction
        psi_colorbar = fig.colorbar(
            psi_image,
            ax=[target_axis, reco_axis],
            fraction=0.046,
            pad=0.04,
        )
        psi_colorbar.set_label(r"$\psi$")

        # Colorbar for the difference
        difference_colorbar = fig.colorbar(
            difference_image,
            ax=difference_axis,
            fraction=0.046,
            pad=0.04,
        )
        difference_colorbar.set_label(
            r"$\Delta\psi=\hat{\psi}-\psi$"
        )

    # =============================================================
    # Multi-channel case
    # =============================================================
    else:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=n_channels,
            figsize=(5 * n_channels, 12),
            squeeze=False,
            constrained_layout=True,
        )

        psi_image = None
        difference_image = None

        for channel in range(n_channels):
            target_axis = axes[0, channel]
            reco_axis = axes[1, channel]
            difference_axis = axes[2, channel]

            # -----------------------------------------------------
            # Target
            # -----------------------------------------------------
            psi_image = target_axis.imshow(
                target_array[channel],
                origin="lower",
                cmap="viridis",
                vmin=psi_vmin,
                vmax=psi_vmax,
            )

            if psi_contour_levels is not None:
                target_axis.contour(
                    r,
                    z,
                    target_array[channel],
                    levels=psi_contour_levels,
                    colors="white",
                    linewidths=0.5,
                )

            target_axis.set_title(
                rf"Input EFIT-$\psi(R,Z)$, channel {channel}"
            )

            # -----------------------------------------------------
            # Reconstruction
            # -----------------------------------------------------
            reco_axis.imshow(
                reco_array[channel],
                origin="lower",
                cmap="viridis",
                vmin=psi_vmin,
                vmax=psi_vmax,
            )

            if psi_contour_levels is not None:
                reco_axis.contour(
                    r,
                    z,
                    reco_array[channel],
                    levels=psi_contour_levels,
                    colors="white",
                    linewidths=0.5,
                )

            reco_axis.set_title(
                rf"Reconstructed EFIT-$\psi(R,Z)$, "
                rf"channel {channel}"
            )

            # -----------------------------------------------------
            # Difference
            # -----------------------------------------------------
            difference_image = difference_axis.imshow(
                difference_array[channel],
                origin="lower",
                cmap="RdBu_r",
                vmin=difference_vmin,
                vmax=difference_vmax,
            )

            difference_axis.contour(
                r,
                z,
                difference_array[channel],
                levels=difference_contour_levels,
                colors="black",
                linewidths=0.5,
            )

            difference_axis.set_title(
                rf"Difference, channel {channel}"
            )

            # Axis formatting
            for axis in (
                target_axis,
                reco_axis,
                difference_axis,
            ):
                axis.set_xlabel(r"$R$ ")
                axis.set_ylabel(r"$Z$ ")
                axis.set_aspect("equal")

        # Shared colorbar for target and reconstruction rows
        psi_colorbar = fig.colorbar(
            psi_image,
            ax=axes[:2, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        psi_colorbar.set_label(r"$\psi$")

        # Shared colorbar for the difference row
        difference_colorbar = fig.colorbar(
            difference_image,
            ax=axes[2, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        difference_colorbar.set_label(
            r"$\Delta\psi=\hat{\psi}-\psi$"
        )

    # -------------------------------------------------------------
    # Save and close the figure
    # -------------------------------------------------------------
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)