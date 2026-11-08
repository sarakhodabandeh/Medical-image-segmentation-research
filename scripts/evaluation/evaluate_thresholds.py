import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from src.models.unet import UNet


CHECKPOINT = (
    ROOT / "results" / "unet_weighted_bce_2_10ep" / "best_model.pt"
)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
def main():
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
        dataset, batch_size=4, shuffle=False, num_workers=0
    )

    # Cache probabilities and targets so each threshold uses
    # exactly the same model predictions.
    records = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images = F.interpolate(
                images,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            masks = F.interpolate(
                masks,
                size=(image_size, image_size),
                mode="nearest",
            )

            logits = model(images)
            probabilities = torch.sigmoid(logits).cpu()
            targets = masks.cpu() >= 0.5

            for i, label in enumerate(batch["label"]):
                records.append({
                    "label": label,
                    "probabilities": probabilities[i, 0],
                    "target": targets[i, 0],
                })

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("\nThreshold results")
    print(
        f"{'Threshold':>9} | {'Lesion Dice':>11} | "
        f"{'Normal FP rate':>14} | {'Mean FP fraction':>16}"
    )

    for threshold in THRESHOLDS:
        dice_values = []
        normal_fp_images = 0
        normal_count = 0
        normal_positive_fractions = []

        for record in records:
            pred = record["probabilities"] >= threshold
            target = record["target"]

            if record["label"] in ("Benign", "Malignant"):
                intersection = (pred & target).sum().item()
                denominator = pred.sum().item() + target.sum().item()

                dice = (
                    2 * intersection / denominator
                    if denominator > 0 else 1.0
                )
                dice_values.append(dice)

            elif record["label"] == "Normal":
                normal_count += 1
                fraction = pred.float().mean().item()
                normal_positive_fractions.append(fraction)

                if pred.any().item():
                    normal_fp_images += 1

        mean_dice = (
            sum(dice_values) / len(dice_values)
            if dice_values else 0.0
        )
        fp_rate = (
            normal_fp_images / normal_count
            if normal_count else 0.0
        )
        mean_fp_fraction = (
            sum(normal_positive_fractions) / len(normal_positive_fractions)
            if normal_positive_fractions else 0.0
        )

        print(
            f"{threshold:9.2f} | {mean_dice:11.4f} | "
            f"{fp_rate:14.4f} | {mean_fp_fraction:16.4f}"
        )


if __name__ == "__main__":
    main()
