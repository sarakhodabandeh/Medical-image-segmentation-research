import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw


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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import (
    resize_with_padding,
    resize_with_constant_padding,
)
from src.models.unet import UNet


DEFAULT_SEEDS = [42, 123, 2025, 7, 99]

# Each image is paired with a seed for which it belongs to the held-out test set.
# These examples were selected from test_per_image_metrics.csv.
SAMPLES = [
    {"filename": "CHVI_005.png", "seed": 123, "reason": "Malignant lesion; Dice=0 in both methods"},
    {"filename": "DAPA_025.png", "seed": 42, "reason": "Benign lesion; Dice=0 in both methods"},
    {"filename": "PLBA_024.png", "seed": 123, "reason": "Benign lesion; high Dice in both methods"},
    {"filename": "FLKA_002.png", "seed": 7, "reason": "Malignant lesion; high Dice in both methods"},
    {"filename": "STSP_022.png", "seed": 123, "reason": "Normal image; high false-positive area"},
]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def preprocess(images, masks, size, padding_mode):
    if padding_mode == "replicate":
        return resize_with_padding(images, masks, size)
    if padding_mode == "constant":
        return resize_with_constant_padding(images, masks, size)
    raise ValueError(f"Unknown padding mode: {padding_mode}")


def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = int(pred.sum()) + int(target.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(pred, target).sum()) / denom


def make_overlay(image, target, prediction):
    """Overlay GT in green, prediction in red, and overlap in yellow."""
    base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    out = base.astype(np.float32).copy()

    gt = target.astype(bool)
    pred = prediction.astype(bool)
    colors = np.zeros_like(out)
    colors[gt] = (0, 220, 0)
    colors[pred] = (255, 40, 40)
    colors[gt & pred] = (255, 220, 0)

    active = gt | pred
    alpha = 0.48
    out[active] = (1 - alpha) * out[active] + alpha * colors[active]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_caption(image, title, subtitle, panel_width=520, panel_height=570):
    canvas = Image.new("RGB", (panel_width, panel_height), "white")
    image = image.resize((panel_width, panel_width), Image.Resampling.BILINEAR)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, panel_width + 8), title, fill="black")
    draw.text((12, panel_width + 30), subtitle, fill="black")
    return canvas


def load_model(checkpoint_path, base_config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", base_config)
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, cfg


def predict(model, image_tensor, mask_tensor, image_size, padding_mode, device, use_amp):
    images = image_tensor.unsqueeze(0).to(device)
    masks = mask_tensor.unsqueeze(0).to(device)
    images, masks = preprocess(images, masks, image_size, padding_mode)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            logits = model(images)
        pred = (torch.sigmoid(logits)[0, 0] >= 0.5).cpu().numpy()
    target = (masks[0, 0] >= 0.5).cpu().numpy()
    display_image = images[0].detach().cpu().permute(1, 2, 0).numpy()
    return display_image, target, pred


def main():
    parser = argparse.ArgumentParser(
        description="Create paired visual comparisons for selected BUS-UCLM test images."
    )
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--splits-dir", default="data/processed/paired_splits")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/test_visualizations")
    parser.add_argument("--metrics-csv", default="results/test_evaluation/test_per_image_metrics.csv")
    args = parser.parse_args()

    with resolve_path(args.config).open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    image_size = int(base_config["data"]["image_size"])
    use_amp = bool(base_config["training"]["use_amp"])
    splits_dir = resolve_path(args.splits_dir)
    results_dir = resolve_path(args.results_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics_path = resolve_path(args.metrics_csv)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Prediction threshold: 0.5 (fixed)")

    summary_rows = []

    for sample_info in SAMPLES:
        filename = sample_info["filename"]
        seed = sample_info["seed"]
        split_csv = splits_dir / f"image_level_seed_{seed}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        dataset = BUSUCLMDataset(split="test", split_path=split_csv)
        indices = [i for i, item in enumerate(dataset.samples) if item["filename"] == filename]
        if len(indices) != 1:
            raise RuntimeError(
                f"Expected exactly one test sample named {filename} for seed {seed}; "
                f"found {len(indices)}."
            )

        sample = dataset[indices[0]]
        label = sample["label"]
        patient_id = sample["patient_id"]
        image_tensor = sample["image"]
        mask_tensor = sample["mask"]

        panels = []
        panel_results = {}

        for split_type in ("image", "patient"):
            checkpoint_path = (
                results_dir / f"paired_full_{split_type}_{seed}" / "best_model.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

            model, checkpoint, cfg = load_model(checkpoint_path, base_config, device)
            padding_mode = checkpoint.get("padding_mode", "replicate")
            display_image, target, prediction = predict(
                model,
                image_tensor,
                mask_tensor,
                image_size,
                padding_mode,
                device,
                use_amp,
            )
            score = dice_score(prediction, target) if label != "Normal" else float("nan")
            fp_area = float(prediction.mean()) if label == "Normal" else float(
                np.logical_and(prediction, ~target).mean()
            )
            panel_results[split_type] = {
                "dice": score,
                "false_positive_area_fraction": fp_area,
                "epoch": checkpoint.get("epoch"),
                "padding_mode": padding_mode,
            }

            # Keep the same image and ground truth for both methods.
            if split_type == "image":
                original_panel = add_caption(
                    Image.fromarray(np.clip(display_image * 255, 0, 255).astype(np.uint8)),
                    "Original image",
                    f"{filename} | {label} | patient {patient_id}",
                )
                gt_overlay = make_overlay(display_image, target, np.zeros_like(target))
                gt_panel = add_caption(
                    gt_overlay,
                    "Ground truth",
                    "Green = lesion mask",
                )
                panels.extend([original_panel, gt_panel])

            pred_overlay = make_overlay(display_image, target, prediction)
            dice_text = f"Dice = {score:.3f}" if np.isfinite(score) else "Dice = N/A (Normal)"
            subtitle = (
                f"{dice_text} | FP area = {fp_area:.3f}"
                if label == "Normal"
                else dice_text
            )
            panels.append(
                add_caption(
                    pred_overlay,
                    f"{split_type.capitalize()}-level prediction",
                    subtitle + " | Green=GT, red=prediction, yellow=overlap",
                )
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Four panels: original, GT, image-level, patient-level.
        panel_width = 520
        panel_height = 570
        montage = Image.new("RGB", (panel_width * 2, panel_height * 2), (235, 235, 235))
        positions = [(0, 0), (panel_width, 0), (0, panel_height), (panel_width, panel_height)]
        for panel, position in zip(panels, positions):
            montage.paste(panel, position)

        safe_name = Path(filename).stem
        output_path = output_dir / f"{safe_name}_seed_{seed}_comparison.png"
        montage.save(output_path)

        summary_rows.append({
            "filename": filename,
            "patient_id": patient_id,
            "label": label,
            "seed": seed,
            "reason_for_selection": sample_info["reason"],
            "image_level_dice": panel_results["image"]["dice"],
            "patient_level_dice": panel_results["patient"]["dice"],
            "image_level_false_positive_area_fraction": (
                panel_results["image"]["false_positive_area_fraction"]
                if label == "Normal" else np.nan
            ),
            "patient_level_false_positive_area_fraction": (
                panel_results["patient"]["false_positive_area_fraction"]
                if label == "Normal" else np.nan
            ),
            "image_level_selected_epoch": panel_results["image"]["epoch"],
            "patient_level_selected_epoch": panel_results["patient"]["epoch"],
            "visualization_file": str(output_path.relative_to(PROJECT_ROOT)),
        })
        print(f"Saved: {output_path}")

    summary_path = output_dir / "visualization_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("\nVisual comparisons completed.")
    print("Output directory:", output_dir)
    print("Summary CSV:", summary_path)


if __name__ == "__main__":
    main()
