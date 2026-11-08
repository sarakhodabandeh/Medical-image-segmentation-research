import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


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


CHECKPOINT_PATH = (
    PROJECT_ROOT / "results" / "unet_baseline" / "best_model.pt"
)


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

    dataset = BUSUCLMDataset(split="val")

    # Prefer examples containing lesions, then include normal scans.
    lesion_indices = [
        i for i, sample in enumerate(dataset.samples)
        if sample["label"] in ("Benign", "Malignant")
    ]
    normal_indices = [
        i for i, sample in enumerate(dataset.samples)
        if sample["label"] == "Normal"
    ]

    selected_indices = (
        lesion_indices[:3] + normal_indices[:1]
    )

    fig, axes = plt.subplots(
        len(selected_indices), 4,
        figsize=(14, 4 * len(selected_indices)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, index in enumerate(selected_indices):
        sample = dataset[index]

        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"][0].numpy()

        image_resized = F.interpolate(
            image,
            size=(config["data"]["image_size"],
                  config["data"]["image_size"]),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            logits = model(image_resized)
            probability = torch.sigmoid(logits)[0, 0]
            prediction = (probability >= 0.5).float().cpu().numpy()

        original_image = sample["image"].permute(1, 2, 0).numpy()

        target_resized = F.interpolate(
            sample["mask"].unsqueeze(0),
            size=(config["data"]["image_size"],
                  config["data"]["image_size"]),
            mode="nearest",
        )[0, 0].numpy()

        image_display = F.interpolate(
            sample["image"].unsqueeze(0),
            size=(config["data"]["image_size"],
                  config["data"]["image_size"]),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).numpy()

        axes[row, 0].imshow(image_display)
        axes[row, 0].set_title(
            f"Image: {sample['filename']} ({sample['label']})"
        )

        axes[row, 1].imshow(target_resized, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground-truth mask")

        axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted mask")

        axes[row, 3].imshow(image_display)
        axes[row, 3].imshow(
            prediction, cmap="Reds", alpha=0.4, vmin=0, vmax=1
        )
        axes[row, 3].set_title("Prediction overlay")

        for col in range(4):
            axes[row, col].axis("off")

        print(
            sample["filename"],
            "| label:", sample["label"],
            "| mean probability:", f"{probability.mean().item():.4f}",
            "| predicted positive fraction:",
            f"{prediction.mean():.4f}",
        )

    plt.show()


if __name__ == "__main__":
    main()
