import sys
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    raise RuntimeError("Could not locate project root.")


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "unet_replicate_pw5_fp30"
    / "epoch_007.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test_evaluation"
    / "preprocessing_comparison"
)

TARGET_FILENAMES = {
    "CAWI_005.png",
    "CAWI_010.png",
    "CODE_000.png",
}

IMAGE_SIZE = 256
THRESHOLD = 0.60


def calculate_dice(prediction, target):
    intersection = (prediction & target).sum().item()
    denominator = prediction.sum().item() + target.sum().item()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = BUSUCLMDataset(split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    with torch.inference_mode():
        for batch in loader:
            filename = batch["filename"][0]

            if filename not in TARGET_FILENAMES:
                continue

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            # Method 1: aspect-ratio-preserving resize + padding.
            padded_image, padded_mask = resize_with_padding(
                image, mask, IMAGE_SIZE
            )

            # Method 2: direct resize, for diagnosis only.
            direct_image = F.interpolate(
                image,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            direct_mask = F.interpolate(
                mask,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest",
            )

            methods = [
                ("padding", padded_image, padded_mask),
                ("direct_resize", direct_image, direct_mask),
            ]

            fig, axes = plt.subplots(
                2, 4, figsize=(14, 7)
            )

            for row_index, (method, img, target_mask) in enumerate(methods):
                probability = torch.sigmoid(model(img))[0, 0].cpu()
                target = target_mask[0, 0].cpu() >= 0.5
                prediction = probability >= THRESHOLD

                dice = calculate_dice(prediction, target)

                print(
                    f"{filename} | {method} | "
                    f"max_prob={probability.max().item():.4f} | "
                    f"pred_pixels={prediction.sum().item()} | "
                    f"target_pixels={target.sum().item()} | "
                    f"dice={dice:.4f}"
                )

                records.append({
                    "filename": filename,
                    "method": method,
                    "max_probability": probability.max().item(),
                    "predicted_pixels": prediction.sum().item(),
                    "target_pixels": target.sum().item(),
                    "dice": dice,
                })

                image_np = (
                    img[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
                )

                axes[row_index, 0].imshow(image_np)
                axes[row_index, 0].set_title(
                    f"{method}: image"
                )

                axes[row_index, 1].imshow(
                    target.numpy(), cmap="gray"
                )
                axes[row_index, 1].set_title("Ground truth")

                axes[row_index, 2].imshow(
                    prediction.numpy(), cmap="gray"
                )
                axes[row_index, 2].set_title(
                    f"Prediction, Dice={dice:.3f}"
                )

                axes[row_index, 3].imshow(
                    probability.numpy(),
                    cmap="hot",
                    vmin=0,
                    vmax=1,
                )
                axes[row_index, 3].set_title(
                    f"Max probability={probability.max().item():.3f}"
                )

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(
                f"{filename} | checkpoint epoch {checkpoint['epoch']} "
                f"| threshold {THRESHOLD}"
            )
            plt.tight_layout()

            fig.savefig(
                OUTPUT_DIR / f"{Path(filename).stem}_comparison.png",
                dpi=130,
                bbox_inches="tight",
            )
            plt.close(fig)

    csv_path = OUTPUT_DIR / "comparison_metrics.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "method",
                "max_probability",
                "predicted_pixels",
                "target_pixels",
                "dice",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nComparison saved to:", OUTPUT_DIR)
    print("Metrics CSV:", csv_path)


if __name__ == "__main__":
    main()
