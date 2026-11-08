import sys
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import ndimage


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
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


CHECKPOINT = (
    ROOT / "results" / "unet_aspect_ratio_pw5" / "best_model.pt"
)

OUTPUT_DIR = ROOT / "results" / "false_positive_components"

THRESHOLDS = [0.60, 0.75]
MIN_COMPONENT_SIZES = [0, 10, 25, 50, 100, 200]
BATCH_SIZE = 4

def remove_small_components(binary_mask, min_size):
    """Remove connected foreground components smaller than min_size."""
    if min_size <= 1:
        return binary_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labels, number = ndimage.label(binary_mask, structure=structure)

    if number == 0:
        return binary_mask

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False

    return keep[labels]


def calculate_dice(pred, target):
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]
    image_size = config["data"]["image_size"]

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BUSUCLMDataset(split="val")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Collect probabilities and targets once.
    records = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images, masks = resize_with_padding(
                images, masks, image_size
            )

            probabilities = torch.sigmoid(model(images)).cpu()
            targets = (masks.cpu() >= 0.5)

            for i, label in enumerate(batch["label"]):
                records.append({
                    "label": label,
                    "probability": probabilities[i, 0].numpy(),
                    "target": targets[i, 0].numpy(),
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for threshold in THRESHOLDS:
        for min_size in MIN_COMPONENT_SIZES:
            lesion_dices = []
            normal_images = 0
            normal_fp_images = 0
            normal_positive_fractions = []
            component_counts = []
            largest_component_sizes = []

            for record in records:
                probability = record["probability"]
                target = record["target"]
                label = record["label"]

                prediction = probability >= threshold

                if min_size > 0:
                    prediction = remove_small_components(
                        prediction, min_size
                    )

                # Component statistics before filtering are useful
                # for understanding the original false positives.
                if label == "Normal":
                    component_labels, number = ndimage.label(
                        probability >= threshold,
                        structure=np.ones((3, 3), dtype=np.uint8),
                    )

                    sizes = np.bincount(component_labels.ravel())[1:]
                    component_counts.append(int(number))
                    largest_component_sizes.append(
                        int(sizes.max()) if len(sizes) else 0
                    )

                    normal_images += 1

                    positive_fraction = float(prediction.mean())
                    normal_positive_fractions.append(
                        positive_fraction
                    )

                    if prediction.any():
                        normal_fp_images += 1

                elif label in ("Benign", "Malignant"):
                    lesion_dices.append(
                        calculate_dice(prediction, target)
                    )

            mean_lesion_dice = (
                float(np.mean(lesion_dices))
                if lesion_dices else 0.0
            )

            normal_fp_rate = (
                normal_fp_images / max(normal_images, 1)
            )

            mean_normal_positive_fraction = (
                float(np.mean(normal_positive_fractions))
                if normal_positive_fractions else 0.0
            )

            row = {
                "threshold": threshold,
                "min_component_pixels": min_size,
                "mean_lesion_dice": mean_lesion_dice,
                "normal_fp_rate": normal_fp_rate,
                "mean_normal_positive_fraction":
                    mean_normal_positive_fraction,
                "mean_normal_component_count":
                    float(np.mean(component_counts))
                    if component_counts else 0.0,
                "mean_largest_normal_component_pixels":
                    float(np.mean(largest_component_sizes))
                    if largest_component_sizes else 0.0,
            }

            rows.append(row)

            print(
                f"\nThreshold={threshold:.2f} | "
                f"Min component size={min_size}"
            )
            print(
                f"  Mean lesion Dice: {mean_lesion_dice:.4f}"
            )
            print(
                f"  Normal image FP rate: {normal_fp_rate:.4f}"
            )
            print(
                "  Mean Normal positive fraction: "
                f"{mean_normal_positive_fraction:.4f}"
            )

            if min_size == 0:
                print(
                    "  Mean Normal connected components: "
                    f"{row['mean_normal_component_count']:.2f}"
                )
                print(
                    "  Mean largest Normal component (pixels): "
                    f"{row['mean_largest_normal_component_pixels']:.1f}"
                )

    csv_path = OUTPUT_DIR / "component_filtering_results.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nCheckpoint:", CHECKPOINT)
    print("Validation images:", len(records))
    print("Results saved to:", csv_path)


if __name__ == "__main__":
    main()
