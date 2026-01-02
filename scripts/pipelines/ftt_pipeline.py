import os
import sys
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader
import csv
from contextlib import nullcontext

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports

# Add the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", ".."
))  # noqa: E402

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Config imports
from scripts.pipelines.configs.ftt_config import (
    SUBSET_OF_SHOTS, OUTPUT_SUB_FOLDER, BATCH_SIZE, NUM_WORKERS,  # REF_FREQ,
    SOURCE_SIGNAL_LIST, LOCAL_FLAG, INACTIVE_TARGETS,
    WINDOW_SEGMENTER_PARAMS, VERBOSE,
    DEFAULT_INPUT_ENCODERS_BY_MOD, DEFAULT_TARGET_ENCODERS_BY_MOD,
    EPOCHS, LR_TRUNK, LR_HEADS, USE_ADAMW, LOSS_SPACE, EARLY_STOP_PATIENCE,
    RUN_TRAINING, RUN_EVALUATION, SAVE_RESULTS,
)

# Common pipeline helpers
from scripts.pipelines.cnn_pipeline import (
    get_train_test_val_shots,
    fit_mean_and_std_for_signal_transform,
    initialize_datasets,
    initialize_dataloaders
)

# Transforms
from scripts.pipelines.utils.utils import ComposeTransforms, collate_fttransform
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import (
    FillProfileWithZerosTransform
)
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import (
    StdScalingTransform
)
# from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import (
#     SamplingToReferenceTimeTransform
# )
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import TruncationTransform
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import WindowSegmenterTransform
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import DropSampleWithNans
from scripts.pipelines.transforms.shot_level_transforms.ftt_transform import FTTransformPrep
from scripts.MAST_tools.MAST_dataset import MastDataset

# Model & registries
from scripts.pipelines.models.ftt_model import MultiModalFTTransformer
from scripts.pipelines.utils.modality_codecs import get_encoder, get_decoder
from scripts.pipelines.utils.ftt_utils import (
    build_registries_from_shapes,
    train_model_per_target_persistent,
    decode_preds_to_native,
)

# ----------------------------------------------------------------------------------------------------------------------
# SET UP DEVICE AND AMP

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16 
elif device.type == "cuda":
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = None
use_amp = (device.type == "cuda") and (AMP_DTYPE is not None)

print(f"device = {device}")
print(f"AMP_DTYPE = {AMP_DTYPE}")


# ----------------------------------------------------------------------------------------------------------------------
def amp_ctx():
    return torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_amp) if device.type == "cuda"\
        else nullcontext()


# ======================================================================================================================
if __name__ == "__main__":

    print(f"\nNumber of available cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # DATA SPLITS & PER-VAR TRANSFORMS
    # ------------------------------------------------------------------------------------------------------------------

    train_shots, test_shots, val_shots = get_train_test_val_shots(max_index=SUBSET_OF_SHOTS)

    # Fit mean and std for signal transformation
    dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
        train_shots=train_shots,
        source_signal_list=SOURCE_SIGNAL_LIST,
        local=LOCAL_FLAG,
        output_sub_dir=OUTPUT_SUB_FOLDER,
        verbose=True
    )

    all_vars = [f"{src}-{sig}" for src, sig in SOURCE_SIGNAL_LIST]
    signal_transform_map = {
        var: ComposeTransforms([
            FillProfileWithZerosTransform(),
            StdScalingTransform(dict_mean[var], dict_std[var]),
            # SamplingToReferenceTimeTransform(REF_FREQ),
        ])
        for var in all_vars
    }

    # ------------------------------------------------------------------------------------------------------------------
    # SHOT-LEVEL TRANSFORM (still using y_key_to_target; identity mapping)
    # ------------------------------------------------------------------------------------------------------------------

    y_key_to_target = {yk: yk for yk in WINDOW_SEGMENTER_PARAMS['y_keys']}

    shot_transform = ComposeTransforms([
        TruncationTransform(),
        WindowSegmenterTransform(**WINDOW_SEGMENTER_PARAMS),
        DropSampleWithNans(verbose=False),
        FTTransformPrep(
            x_keys=WINDOW_SEGMENTER_PARAMS['x_keys'],
            y_key_to_target=y_key_to_target,  # ← you said you don't have the y_keys version
            ensure_3d_inputs=True,
            ensure_3d_targets=True,
        )
    ])

    # ------------------------------------------------------------------------------------------------------------------
    # DATASETS and DATALOADERS
    # ------------------------------------------------------------------------------------------------------------------

    datasets = initialize_datasets(
        sources_and_signals=SOURCE_SIGNAL_LIST,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=LOCAL_FLAG,
        verbose=False
    )

    dataloaders = initialize_dataloaders(
        datasets=datasets,
        collate_function=collate_fttransform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=False,
        verbose=False,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    print(f'\nlen(train_loader) = {len(train_loader)}')

    # ------------------------------------------------------------------------------------------------------------------
    # SHAPES FROM FIRST WINDOW OF FIRST TRAIN SHOT (no loader)
    # ------------------------------------------------------------------------------------------------------------------

    windows = datasets["train"][0]  # list of (Xs, names, y_native)
    if not windows:
        raise RuntimeError("First train shot produced no windows; try a different shot or adjust window params.")
    sample = windows[0]
    Xs, names, y_native = sample
    input_shapes = [tuple(np.asarray(x).shape) for x in Xs]  # noqa

    # Targets: dict of 3D arrays (C,H,T) because ensure_3d=True
    # target_order = list(names)
    target_order = [n for n in names if n not in INACTIVE_TARGETS]
    target_shapes = [tuple(np.asarray(y_native[n]).shape) for n in target_order]  # noqa

    # ------------------------------------------------------------------------------------------------------------------
    # REGISTRIES (auto modality + defaults per modality)
    # ------------------------------------------------------------------------------------------------------------------

    input_registry, target_registry = build_registries_from_shapes(
        input_names=WINDOW_SEGMENTER_PARAMS['x_keys'],
        input_shapes=input_shapes,
        target_names=target_order,
        target_shapes=target_shapes,
        get_encoder=get_encoder,
        get_decoder=get_decoder,
        default_input_by_modality=DEFAULT_INPUT_ENCODERS_BY_MOD,
        default_target_by_modality=DEFAULT_TARGET_ENCODERS_BY_MOD,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # MODEL INIT
    # ------------------------------------------------------------------------------------------------------------------

    model = MultiModalFTTransformer(
        input_registry=input_registry,
        target_registry=target_registry,
        device=device,
        verbose=VERBOSE
    )

    # ------------------------------------------------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------------------------------------------------

    if RUN_TRAINING:
        print(f'\nTraining...')

        history = train_model_per_target_persistent(
            model,
            train_loader,
            val_loader,
            registry=target_registry,  # loss weights, decoder/meta come from here
            epochs=EPOCHS,
            lr_trunk=LR_TRUNK,
            lr_heads=LR_HEADS,
            wd_trunk=1e-4,
            wd_heads=0.0,
            warmup_steps=0,
            use_adamw=USE_ADAMW,
            grad_accum_steps=1,
            patience=EARLY_STOP_PATIENCE,
            min_delta=0.0,
            restore_best=True,
            loss_space=LOSS_SPACE,  # "pred" or "native"
            verbose=VERBOSE,
        )

        print("\nTraining done.")
        print("Last epoch train/val:", history["train_loss"][-1], history["val_loss"][-1])

        if SAVE_RESULTS:
            out_dir = os.path.join("output", OUTPUT_SUB_FOLDER)
            os.makedirs(out_dir, exist_ok=True)

            # Save the best model weights (trainer restored them into `model`)
            best_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best model to: {best_path}")

    # ------------------------------------------------------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------------------------------------------------------

    if RUN_EVALUATION:
        print(f'\nEvaluation...')

        # Restore best weights if you saved them
        best_path = os.path.join("output", OUTPUT_SUB_FOLDER, "best_model.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        # Decide target order for reporting (use the same order you trained with)
        target_order_eval = [n for n in WINDOW_SEGMENTER_PARAMS['y_keys'] if n not in INACTIVE_TARGETS]

        # CSV setup (only if saving)
        writer = None
        f_csv = None
        csv_path = None
        if SAVE_RESULTS:
            out_dir = os.path.join("output", OUTPUT_SUB_FOLDER)
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "test_loss_per_target.csv")
            f_csv = open(csv_path, "w", newline="")
            writer = csv.writer(f_csv)
            writer.writerow(["shot_id"] + [f"MSE_{n}" for n in target_order_eval])

        try:
            # Loop shots one by one (exactly like the CNN pipeline)
            for shot_id in test_shots:
                print(f"Evaluating shot {shot_id}")

                test_shot_dataset = MastDataset(
                    local=LOCAL_FLAG,
                    shots_list=[shot_id],
                    source_signal_list=SOURCE_SIGNAL_LIST,
                    signal_level_transform_map=signal_transform_map,
                    shot_level_transform=shot_transform,
                )

                # empty-window guard (some shots may produce no windows after transforms)
                if len(test_shot_dataset[0]) == 0:
                    print(f"Shot {shot_id} not run properly, likely empty slice")
                    continue

                test_shot_loader = DataLoader(
                    dataset=test_shot_dataset,
                    batch_size=1,  # one window per sample; we’ll average across all windows
                    num_workers=NUM_WORKERS,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fttransform,
                    pin_memory=False,
                )

                # accumulators (sample-weighted, per target)
                sum_mse_per_target = {n: 0.0 for n in target_order_eval}
                count_per_target = {n: 0 for n in target_order_eval}

                with torch.no_grad():
                    for X_batch, active_targets, y_native in test_shot_loader:
                        # Forward under autocast (bf16/fp16 on CUDA)
                        with amp_ctx():
                            out = model(
                                X_batch,
                                active_targets=active_targets,
                                Y_for_meta=y_native,
                                require_decoded=False,
                            )

                        preds = out["preds"]
                        # Decoding in fp32 on model/device
                        y_pred_native = decode_preds_to_native(model, preds, y_native)

                        # Fast paths: batch size & device
                        B = len(X_batch)              # cheaper than inspecting tensors
                        pred_device = device          # already known / consistent

                        # Pre-move y_native once (only those we’ll actually score)
                        present = set(y_pred_native.keys())
                        y_native_fp32 = {
                            n: t.to(dtype=torch.float32, device=pred_device)
                            for n, t in y_native.items() if n in present
                        }

                        for name in target_order_eval:
                            yp = y_pred_native.get(name)
                            yt = y_native_fp32.get(name)
                            if yp is None or yt is None:
                                continue

                            # yp should already be fp32 on device; cast if you aren't 100% sure
                            diff2 = (yp.float() - yt) ** 2
                            per_sample_mse = diff2.reshape(B, -1).mean(dim=1)
                            batch_mean_mse = float(per_sample_mse.mean())

                            sum_mse_per_target[name] += batch_mean_mse * B
                            count_per_target[name] += B

                # Final per-target MSE (sample-weighted); keep order in target_order_eval
                avg_mse_per_target = [
                    (sum_mse_per_target[n] / max(1, count_per_target[n])) for n in target_order_eval
                ]
                print(f"Shot {shot_id} avg MSE per target:", [round(v, 6) for v in avg_mse_per_target])

                if SAVE_RESULTS and writer is not None:
                    writer.writerow([shot_id] + [float(v) for v in avg_mse_per_target])
                    f_csv.flush()

        finally:
            if f_csv is not None:
                f_csv.close()
                print(f"Saved CSV to: {csv_path}")

    # ------------------------------------------------------------------------------------------------------------------
