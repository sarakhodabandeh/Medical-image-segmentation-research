import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.models.unet import UNet


CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "unet_weighted_bce_2_10ep"
    / "best_model.pt"
)


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
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    shown_lesions = 0
    shown_normal = 0

    with torch.no_grad():
        for batch in loader:
            label = batch["label"][0]

            image = batch["image"].to(device)
            target = batch["mask"].to(device)

            image = F.interpolate(
                image,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

            target = F.interpolate(
                target,
                size=(image_size, image_size),
                mode="nearest",
            )

            probability = torch.sigmoid(model(image))[0, 0].cpu()
            target = target[0, 0].cpu() >= 0.5
            original = image[0].cpu().permute(1, 2, 0).numpy()

            prediction = probability >= 0.5

            if label in ("Benign", "Malignant"):
                if shown_lesions >= 4:
                    continue
                shown_lesions += 1
            elif label == "Normal":
                if shown_normal >= 4:
                    continue
                shown_normal += 1
            else:
                continue

            intersection = (prediction & target).sum().item()
            denominator = prediction.sum().item() + target.sum().item()
            dice = (
                2 * intersection / denominator
                if denominator > 0
                else 1.0
            )

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(original.clip(0, 1))
            axes[0].set_title(f"Original: {label}")

            axes[1].imshow(target, cmap="gray")
            axes[1].set_title("Ground truth")

            axes[2].imshow(prediction, cmap="gray")
            axes[2].set_title("Prediction (0.50)")

            axes[3].imshow(probability, cmap="hot", vmin=0, vmax=1)
            axes[3].set_title("Probability map")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"Dice: {dice:.3f} | "
                f"Predicted area: {prediction.float().mean().item():.3f}"
            )
            plt.tight_layout()
            plt.show()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Lesion examples displayed: {shown_lesions}")
    print(f"Normal examples displayed: {shown_normal}")


if __name__ == "__main__":
    main()
