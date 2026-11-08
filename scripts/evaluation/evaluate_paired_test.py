import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import resize_with_padding, resize_with_constant_padding
from src.models.unet import UNet


def find_project_root():
    current = Path(__file__).resolve().parent

    for directory in (current, *current.parents):
        if (
            (directory / "configs").is_dir()
            and (directory / "src").is_dir()
            and (directory / "requirements.txt").is_file()
        ):
            return directory

    raise RuntimeError(
        "Could not locate project root. "
        "Expected configs/, src/, and requirements.txt."
    )


PROJECT_ROOT = find_project_root()
DEFAULT_SEEDS = [42, 123, 2025, 7, 99]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path



def preprocess(images, masks, size, padding_mode):
    if padding_mode == "replicate":
        return resize_with_padding(images, masks, size)
    if padding_mode == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding mode: {padding_mode}")


def ratio(n, d):
    return float(n / d) if d else float("nan")


def evaluate_model(model, loader, device, image_size, padding_mode, use_amp, run_name):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = preprocess(images, masks, image_size, padding_mode)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(images)

            predictions = torch.sigmoid(logits) >= 0.5
            targets = masks >= 0.5

            for i in range(images.shape[0]):
                pred = predictions[i, 0]
                target = targets[i, 0]
                label = batch["label"][i]
                tp = int((pred & target).sum().item())
                fp = int((pred & ~target).sum().item())
                fn = int((~pred & target).sum().item())
                tn = int((~pred & ~target).sum().item())
                total_pixels = int(pred.numel())

                is_lesion = label in ("Benign", "Malignant")
                rows.append({
                    "run": run_name,
                    "filename": batch["filename"][i],
                    "patient_id": batch["patient_id"][i],
                    "label": label,
                    "tp_pixels": tp,
                    "fp_pixels": fp,
                    "fn_pixels": fn,
                    "tn_pixels": tn,
                    "dice": ratio(2 * tp, 2 * tp + fp + fn) if is_lesion else np.nan,
                    "iou": ratio(tp, tp + fp + fn) if is_lesion else np.nan,
                    "pixel_recall": ratio(tp, tp + fn) if is_lesion else np.nan,
                    "pixel_precision": ratio(tp, tp + fp) if is_lesion else np.nan,
                    "false_positive_area_fraction": ratio(fp, total_pixels),
                    "normal_false_positive_image": (
                        int(bool(pred.any().item())) if label == "Normal" else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def summarize(frame, seed, split_type):
    lesions = frame[frame.label.isin(["Benign", "Malignant"])]
    normals = frame[frame.label == "Normal"]
    result = {
        "seed": seed,
        "split_type": split_type,
        "n_test_images": len(frame),
        "n_lesion_images": len(lesions),
        "n_normal_images": len(normals),
        "lesion_macro_dice": lesions.dice.mean(),
        "lesion_macro_iou": lesions.iou.mean(),
        "lesion_macro_pixel_recall": lesions.pixel_recall.mean(),
        "normal_false_positive_image_rate": normals.normal_false_positive_image.mean(),
        "normal_mean_false_positive_area_fraction": normals.false_positive_area_fraction.mean(),
        "normal_pixel_false_positive_rate": ratio(
            normals.fp_pixels.sum(),
            normals.fp_pixels.sum() + normals.tn_pixels.sum(),
        ),
    }
    for label in ("Benign", "Malignant"):
        subset = lesions[lesions.label == label]
        key = label.lower()
        result[f"{key}_n"] = len(subset)
        result[f"{key}_dice"] = subset.dice.mean()
        result[f"{key}_iou"] = subset.iou.mean()
        result[f"{key}_pixel_recall"] = subset.pixel_recall.mean()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate paired BUS-UCLM U-Net checkpoints on the held-out test sets."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--splits-dir", default="data/processed/paired_splits")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/test_evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    image_size = int(base_config["data"]["image_size"])
    batch_size = args.batch_size or int(base_config["data"]["batch_size"])
    num_workers = int(base_config["data"]["num_workers"])
    use_amp = bool(base_config["training"]["use_amp"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_dir = resolve_path(args.splits_dir)
    results_dir = resolve_path(args.results_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Threshold: 0.5 (fixed; no test-set tuning)")

    image_frames, summaries = [], []
    for seed in args.seeds:
        for split_type in ("image", "patient"):
            split_csv = splits_dir / f"{split_type}_level_seed_{seed}.csv"
            checkpoint_path = results_dir / f"paired_full_{split_type}_{seed}" / "best_model.pt"
            if not split_csv.exists():
                raise FileNotFoundError(f"Missing split CSV: {split_csv}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint_path}. "
                    "Check or rename the run folder to paired_full_<image|patient>_<seed>."
                )

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg = checkpoint.get("config", base_config)
            checkpoint_seed = int(checkpoint.get("seed", seed))
            if checkpoint_seed != seed:
                raise ValueError(f"Seed mismatch in {checkpoint_path}: {checkpoint_seed}")
            padding_mode = checkpoint.get("padding_mode", "replicate")

            dataset = BUSUCLMDataset(split="test", split_path=split_csv)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            model = UNet(
                in_channels=cfg["model"]["in_channels"],
                out_channels=cfg["model"]["out_channels"],
                base_channels=cfg["model"]["base_channels"],
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

            run_name = f"{split_type}_level_seed_{seed}"
            print(
                f"Evaluating {run_name}: {len(dataset)} test images; "
                f"selected epoch={checkpoint.get('epoch')}; padding={padding_mode}"
            )
            per_image = evaluate_model(
                model, loader, device, image_size, padding_mode, use_amp, run_name
            )
            per_image["seed"] = seed
            per_image["split_type"] = split_type
            image_frames.append(per_image)
            summaries.append(summarize(per_image, seed, split_type))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    all_images = pd.concat(image_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    all_images.to_csv(output_dir / "test_per_image_metrics.csv", index=False)
    summary.to_csv(output_dir / "test_summary_by_run.csv", index=False)

    metrics = [
        "lesion_macro_dice",
        "lesion_macro_iou",
        "lesion_macro_pixel_recall",
        "normal_false_positive_image_rate",
        "normal_mean_false_positive_area_fraction",
        "normal_pixel_false_positive_rate",
    ]
    wide = summary.pivot(index="seed", columns="split_type", values=metrics)
    paired = pd.DataFrame(index=wide.index)
    for metric in metrics:
        paired[f"{metric}_image_level"] = wide[(metric, "image")]
        paired[f"{metric}_patient_level"] = wide[(metric, "patient")]
        paired[f"{metric}_difference_patient_minus_image"] = (
            wide[(metric, "patient")] - wide[(metric, "image")]
        )
    paired.reset_index().to_csv(output_dir / "test_paired_comparison.csv", index=False)

    print("\nEvaluation completed. Files saved in:", output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
