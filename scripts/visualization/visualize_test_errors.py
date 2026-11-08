import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


ROOT = PROJECT_ROOT

CHECKPOINT = (
    ROOT / "results" / "unet_replicate_pw5_fp30" / "epoch_007.pt"
)

THRESHOLD = 0.60
NUM_WORST_IMAGES = 10

CSV_PATH = (
    ROOT / "results" / "test_evaluation" / "test_per_image.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "test_evaluation" / "worst_case_visualizations"
)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Load the ten worst lesion cases from the existing test results.
    results = pd.read_csv(CSV_PATH)

    lesion_results = results[
        results["label"].isin(["Benign", "Malignant"])
    ].copy()

    worst_cases = (
        lesion_results
        .sort_values("dice", ascending=True)
        .head(NUM_WORST_IMAGES)
    )

    filenames = set(worst_cases["filename"].astype(str))

    print("Selected test cases:")
    print(
        worst_cases[
            ["filename", "patient_id", "label", "dice"]
        ].to_string(index=False)
    )

    # Load the fixed checkpoint.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]
    size = config["data"]["image_size"]

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

    saved = 0

    with torch.no_grad():
        for batch in loader:
            filename = str(batch["filename"][0])

            if filename not in filenames:
                continue

            label = batch["label"][0]
            patient_id = str(batch["patient_id"][0])

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images, masks = resize_with_padding(
                images, masks, size
            )

            probability = torch.sigmoid(model(images))[0, 0].cpu()
            target = masks[0, 0].cpu() >= 0.5
            prediction = probability >= THRESHOLD
            max_value = probability.max().item()
            max_position = torch.nonzero(
                probability == probability.max(),
                as_tuple=False
            )[0].tolist()

            print(
                f"{filename} | "
                f"max probability={max_value:.4f} | "
                f"target pixels={target.sum().item()} | "
                f"predicted pixels={prediction.sum().item()} | "
                f"max position (y, x)={max_position}"
            )

            intersection = (prediction & target).sum().item()
            pred_pixels = prediction.sum().item()
            target_pixels = target.sum().item()

            denominator = pred_pixels + target_pixels
            dice = (
                2 * intersection / denominator
                if denominator > 0
                else 1.0
            )

            image = images[0].cpu().permute(1, 2, 0).numpy()

            # Green = true positive, red = false positive,
            # blue = false negative.
            overlay = torch.zeros(
                (*target.shape, 3), dtype=torch.float32
            )
            overlay[..., 1] = (prediction & target).float()
            overlay[..., 0] = (prediction & ~target).float()
            overlay[..., 2] = (~prediction & target).float()

            fig, axes = plt.subplots(1, 5, figsize=(18, 4))

            axes[0].imshow(image.clip(0, 1))
            axes[0].set_title(f"Image: {label}")

            axes[1].imshow(target, cmap="gray")
            axes[1].set_title("Ground truth")

            axes[2].imshow(prediction, cmap="gray")
            axes[2].set_title(f"Prediction @ {THRESHOLD}")

            axes[3].imshow(
                probability, cmap="hot", vmin=0, vmax=1
            )
            axes[3].set_title("Probability")

            axes[4].imshow(overlay.numpy())
            axes[4].set_title("Green=TP, Red=FP, Blue=FN")

            fig.suptitle(
                f"{filename} | Patient={patient_id} | "
                f"Dice={dice:.3f} | "
                f"Pred={pred_pixels} | Target={target_pixels}"
            )

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()

            output_name = Path(filename).stem + "_error.png"
            fig.savefig(
                OUTPUT_DIR / output_name,
                dpi=140,
                bbox_inches="tight",
            )
            plt.close(fig)

            saved += 1

    print(f"\nCheckpoint epoch: {checkpoint['epoch']}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Images saved: {saved}")
    print(f"Output folder: {OUTPUT_DIR}")

    if saved != len(filenames):
        print(
            "Warning: Some selected filenames were not found in "
            "the test dataset. Check the filenames and CSV."
        )


if __name__ == "__main__":
    main()
