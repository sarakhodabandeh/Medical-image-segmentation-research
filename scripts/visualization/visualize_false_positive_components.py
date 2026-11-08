import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
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


CHECKPOINT = (
    PROJECT_ROOT / "results" / "unet_aspect_ratio_pw5" / "best_model.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "false_positive_visualization"

THRESHOLDS = [0.60, 0.75]
BATCH_SIZE = 4
MAX_IMAGES_PER_CLASS = 12
MIN_COMPONENT_SIZE_TO_LABEL = 100


def get_components(binary_mask):
    """Return connected-component labels and component sizes."""
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, number = ndimage.label(
        binary_mask,
        structure=structure,
    )

    sizes = np.bincount(labels.ravel())[1:]
    return labels, number, sizes


def prepare_image(image_tensor):
    """Convert a [C,H,W] tensor to an image for plotting."""
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()

    # The dataset images are expected to be in [0,1].
    return np.clip(image, 0, 1)


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_counts = {
        "Normal": 0,
        "Benign": 0,
        "Malignant": 0,
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images, masks = resize_with_padding(
                images,
                masks,
                image_size,
            )

            probabilities = torch.sigmoid(model(images)).cpu()
            targets = (masks.cpu() >= 0.5)

            for i, label in enumerate(batch["label"]):
                if label not in saved_counts:
                    continue

                if saved_counts[label] >= MAX_IMAGES_PER_CLASS:
                    continue

                image = prepare_image(images[i])
                probability = probabilities[i, 0].numpy()
                target = targets[i, 0].numpy()

                for threshold in THRESHOLDS:
                    prediction = probability >= threshold

                    component_labels, number, sizes = get_components(
                        prediction
                    )

                    # Different colors for different connected regions.
                    component_map = np.zeros(
                        (*prediction.shape, 3),
                        dtype=float,
                    )

                    rng = np.random.default_rng(42)

                    for component_id in range(1, number + 1):
                        size = int(sizes[component_id - 1])

                        if size < MIN_COMPONENT_SIZE_TO_LABEL:
                            continue

                        color = rng.uniform(0.2, 1.0, size=3)
                        component_map[
                            component_labels == component_id
                        ] = color

                    fig, axes = plt.subplots(
                        1, 5, figsize=(18, 4)
                    )

                    axes[0].imshow(image)
                    axes[0].set_title(f"{label}: image")

                    axes[1].imshow(
                        target,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                    axes[1].set_title("Ground truth")

                    axes[2].imshow(
                        probability,
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                    )
                    axes[2].set_title("Probability")

                    axes[3].imshow(
                        prediction,
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                    axes[3].set_title(
                        f"Prediction, threshold={threshold}"
                    )

                    axes[4].imshow(image)
                    axes[4].imshow(
                        component_map,
                        alpha=0.8,
                    )
                    axes[4].set_title(
                        "Connected components"
                    )

                    # Label the larger components with their pixel sizes.
                    for component_id in range(1, number + 1):
                        size = int(sizes[component_id - 1])

                        if size < MIN_COMPONENT_SIZE_TO_LABEL:
                            continue

                        ys, xs = np.where(
                            component_labels == component_id
                        )

                        if len(xs) == 0:
                            continue

                        x_center = float(xs.mean())
                        y_center = float(ys.mean())

                        axes[4].text(
                            x_center,
                            y_center,
                            str(size),
                            color="white",
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox={
                                "facecolor": "black",
                                "alpha": 0.7,
                                "pad": 1,
                            },
                        )

                    for ax in axes:
                        ax.axis("off")

                    fig.tight_layout()

                    filename = (
                        f"{label.lower()}_"
                        f"{saved_counts[label] + 1:02d}_"
                        f"threshold_{threshold:.2f}.png"
                    )

                    fig.savefig(
                        OUTPUT_DIR / filename,
                        dpi=130,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

                    if threshold == THRESHOLDS[0]:
                        print(
                            f"{label} image "
                            f"{saved_counts[label] + 1}: "
                            f"{number} components at threshold "
                            f"{threshold:.2f}"
                        )

                saved_counts[label] += 1

    print("\nCheckpoint:", CHECKPOINT)
    print("Saved images:", saved_counts)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
