import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

CHECKPOINTS = {
    "epoch_005": (
        ROOT / "results" / "unet_replicate_pw5_fp30" / "epoch_005.pt"
    ),
    "epoch_007": (
        ROOT / "results" / "unet_replicate_pw5_fp30" / "epoch_007.pt"
    ),
}

THRESHOLD = 0.50
MAX_LESION_EXAMPLES = 4
MAX_NORMAL_EXAMPLES = 4


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = BUSUCLMDataset(split="val")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    output_root = ROOT / "results" / "visual_comparison_pw5_fp30"
    output_root.mkdir(parents=True, exist_ok=True)

    for model_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path,
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

        model_output = output_root / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        shown_lesions = 0
        shown_normal = 0

        with torch.no_grad():
            for batch in loader:
                label = batch["label"][0]

                if label in ("Benign", "Malignant"):
                    if shown_lesions >= MAX_LESION_EXAMPLES:
                        continue
                elif label == "Normal":
                    if shown_normal >= MAX_NORMAL_EXAMPLES:
                        continue
                else:
                    continue

                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                images, masks = resize_with_padding(images, masks, size)

                probability = torch.sigmoid(model(images))[0, 0].cpu()
                target = masks[0, 0].cpu() >= 0.5
                prediction = probability >= THRESHOLD

                if label in ("Benign", "Malignant"):
                    shown_lesions += 1
                else:
                    shown_normal += 1

                intersection = (prediction & target).sum().item()
                denominator = prediction.sum().item() + target.sum().item()
                dice = (
                    2 * intersection / denominator
                    if denominator > 0
                    else (1.0 if not target.any().item() else 0.0)
                )

                image = images[0].cpu().permute(1, 2, 0).numpy()

                fig, axes = plt.subplots(1, 5, figsize=(18, 4))
                axes[0].imshow(image.clip(0, 1))
                axes[0].set_title(f"Original: {label}")

                axes[1].imshow(target, cmap="gray")
                axes[1].set_title("Ground truth")

                axes[2].imshow(prediction, cmap="gray")
                axes[2].set_title(f"Prediction @ {THRESHOLD}")

                axes[3].imshow(probability, cmap="hot", vmin=0, vmax=1)
                axes[3].set_title("Probability")

                # Green = true positive, red = false positive,
                # blue = false negative pixels.
                overlay = torch.zeros(
                    (*target.shape, 3), dtype=torch.float32
                )
                overlay[..., 1] = (prediction & target).float()
                overlay[..., 0] = (prediction & ~target).float()
                overlay[..., 2] = (~prediction & target).float()

                axes[4].imshow(overlay.numpy())
                axes[4].set_title("Green=TP, Red=FP, Blue=FN")

                fig.suptitle(
                    f"{model_name} | Dice={dice:.3f} | "
                    f"Predicted fraction="
                    f"{prediction.float().mean().item():.3f}"
                )
                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()

                filename = (
                    f"{shown_lesions:02d}_lesion.png"
                    if label in ("Benign", "Malignant")
                    else f"{shown_normal:02d}_normal.png"
                )
                fig.savefig(
                    model_output / filename,
                    dpi=120,
                    bbox_inches="tight",
                )
                plt.close(fig)

        print(f"\nModel: {model_name}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Lesion examples saved: {shown_lesions}")
        print(f"Normal examples saved: {shown_normal}")
        print(f"Images saved to: {model_output}")

    print(f"\nAll outputs are under: {output_root}")


if __name__ == "__main__":
    main()
