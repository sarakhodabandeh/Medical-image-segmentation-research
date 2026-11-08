import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


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


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)
from src.models.unet import UNet


THRESHOLDS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]


def project_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
def preprocess(images, masks, size, padding):
    if padding == "replicate":
        return resize_with_padding(images, masks, size)
    if padding == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding mode: {padding}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument(
        "--split-csv",
        default="data/processed/paired_splits/patient_level_seed_99.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="results/diagnostic_patient_99/epoch_checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        default="results/diagnostic_patient_99/probability_analysis",
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()

    with project_path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    use_amp = bool(config["training"]["use_amp"])

    dataset = BUSUCLMDataset(
        split="val",
        split_path=project_path(args.split_csv),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_rows = []
    summary_rows = []

    print("Device:", device)
    print("Validation images:", len(dataset))
    print("Thresholds:", THRESHOLDS)

    for epoch in args.epochs:
        ckpt_path = project_path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        padding = ckpt.get("padding_mode", "replicate")
        model = UNet(
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
            base_channels=cfg["model"]["base_channels"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rows = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                imgs, masks = preprocess(imgs, masks, image_size, padding)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp and device.type == "cuda",
                ):
                    logits = model(imgs)
                probs = torch.sigmoid(logits).float()

                for i in range(probs.shape[0]):
                    p = probs[i, 0].detach().cpu().numpy()
                    target = (masks[i, 0].detach().cpu().numpy() >= 0.5)
                    label = batch["label"][i]
                    lesion = label in ("Benign", "Malignant")

                    row = {
                        "epoch": epoch,
                        "filename": batch["filename"][i],
                        "patient_id": batch["patient_id"][i],
                        "label": label,
                        "prob_mean": float(p.mean()),
                        "prob_std": float(p.std()),
                        "prob_min": float(p.min()),
                        "prob_max": float(p.max()),
                        "prob_q50": float(np.quantile(p, 0.50)),
                        "prob_q90": float(np.quantile(p, 0.90)),
                        "prob_q95": float(np.quantile(p, 0.95)),
                        "prob_q99": float(np.quantile(p, 0.99)),
                        "gt_area_fraction": float(target.mean()),
                    }
                    for t in THRESHOLDS:
                        pred = p >= t
                        row[f"pred_area_fraction_at_{t:.2f}"] = float(pred.mean())
                        if lesion:
                            tp = int(np.logical_and(pred, target).sum())
                            denom = int(pred.sum() + target.sum())
                            row[f"dice_at_{t:.2f}"] = (
                                2 * tp / denom if denom else 1.0
                            )
                        else:
                            row[f"normal_any_fp_at_{t:.2f}"] = int(pred.any())

                    rows.append(row)

        frame = pd.DataFrame(rows)
        per_image_rows.append(frame)

        for label in ["Benign", "Malignant", "Normal"]:
            subset = frame[frame.label == label]
            if subset.empty:
                continue
            summary = {
                "epoch": epoch,
                "label": label,
                "n_images": len(subset),
                "mean_probability": subset.prob_mean.mean(),
                "median_probability_q99": subset.prob_q99.median(),
                "mean_predicted_area_at_0.50": subset["pred_area_fraction_at_0.50"].mean(),
            }
            if label == "Normal":
                for t in THRESHOLDS:
                    summary[f"normal_image_fp_rate_at_{t:.2f}"] = (
                        subset[f"normal_any_fp_at_{t:.2f}"].mean()
                    )
            else:
                for t in THRESHOLDS:
                    summary[f"macro_dice_at_{t:.2f}"] = subset[f"dice_at_{t:.2f}"].mean()
            summary_rows.append(summary)

        print(f"Epoch {epoch:02d} analyzed.")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_images = pd.concat(per_image_rows, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)
    all_images.to_csv(output_dir / "validation_probabilities_per_image.csv", index=False)
    summaries.to_csv(output_dir / "validation_probability_summary.csv", index=False)

    print("\nSaved:")
    print(output_dir / "validation_probabilities_per_image.csv")
    print(output_dir / "validation_probability_summary.csv")
    print("\nSummary:")
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
