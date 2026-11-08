import sys
import csv
from pathlib import Path

import torch
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
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "unet_replicate_pw5_fp30"
    / "epoch_007.pt"
)

OUTPUT_DIR = ROOT / "results" / "test_evaluation"
THRESHOLD = 0.60
IMAGE_SIZE = 256
BATCH_SIZE = 4

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Threshold:", THRESHOLD)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model = UNet(
        in_channels=model_config.get("in_channels", 3),
        out_channels=model_config.get("out_channels", 1),
        base_channels=model_config.get("base_channels", 16),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BUSUCLMDataset(split="test")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Test images:", len(dataset))

    # Store per-image metrics so results can be analyzed by label.
    records = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images, masks = resize_with_padding(
                images, masks, IMAGE_SIZE
            )

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= THRESHOLD
            targets = masks >= 0.5

            for i in range(images.shape[0]):
                pred = predictions[i, 0]
                target = targets[i, 0]

                tp = (pred & target).sum().item()
                fp = (pred & ~target).sum().item()
                fn = (~pred & target).sum().item()

                denominator = 2 * tp + fp + fn
                dice = (
                    2 * tp / denominator
                    if denominator > 0
                    else 1.0
                )

                records.append(
                    {
                        "filename": batch["filename"][i],
                        "patient_id": batch["patient_id"][i],
                        "label": batch["label"][i],
                        "dice": dice,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "predicted_pixels": int(pred.sum().item()),
                        "target_pixels": int(target.sum().item()),
                    }
                )

    if not records:
        raise RuntimeError("No test predictions were generated.")

    # Summarize each label separately.
    summary = []

    for label in ("Benign", "Malignant", "Normal"):
        label_records = [
            record for record in records
            if record["label"] == label
        ]

        if not label_records:
            continue

        tp = sum(record["tp"] for record in label_records)
        fp = sum(record["fp"] for record in label_records)
        fn = sum(record["fn"] for record in label_records)

        pixel_dice = (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn > 0
            else 1.0
        )
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        row = {
            "label": label,
            "num_images": len(label_records),
            "mean_image_dice": sum(
                record["dice"] for record in label_records
            ) / len(label_records),
            "pixel_dice": pixel_dice,
            "pixel_precision": precision,
            "pixel_recall": recall,
            "normal_fp_rate": "",
        }

        if label == "Normal":
            row["normal_fp_rate"] = sum(
                record["predicted_pixels"] > 0
                for record in label_records
            ) / len(label_records)

        summary.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_image_path = OUTPUT_DIR / "test_per_image.csv"
    summary_path = OUTPUT_DIR / "test_summary_by_label.csv"

    with per_image_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nTest results by label:")
    for row in summary:
        print(
            f"{row['label']:10s} | "
            f"n={row['num_images']:3d} | "
            f"Mean image Dice={row['mean_image_dice']:.4f} | "
            f"Pixel Dice={row['pixel_dice']:.4f} | "
            f"Precision={row['pixel_precision']:.4f} | "
            f"Recall={row['pixel_recall']:.4f} | "
            f"Normal FP rate={row['normal_fp_rate']}"
        )

    print("\nPer-image results saved to:", per_image_path)
    print("Label summary saved to:", summary_path)


if __name__ == "__main__":
    main()
