import sys
from pathlib import Path

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


ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.datasets.bus_uclm import BUSUCLMDataset
from src.datasets.preprocessing import resize_with_padding
from src.models.unet import UNet


CHECKPOINT = (
    ROOT / "results" / "unet_aspect_ratio_pw5" / "best_model.pt"
)

OUTPUT_DIR = ROOT / "results" / "probability_analysis"

THRESHOLD = 0.60
BATCH_SIZE = 4
MAX_EXAMPLES_PER_CLASS = 4


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
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

    # Probability values from the three groups.
    normal_probs = []
    lesion_probs = []
    lesion_background_probs = []

    # Per-image statistics.
    normal_image_stats = []
    lesion_image_stats = []

    # Store a few examples for visualization.
    examples = {
        "Normal": [],
        "Benign": [],
        "Malignant": [],
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
            targets = masks.cpu() >= 0.5

            for i, label in enumerate(batch["label"]):
                image = images[i].cpu()
                prob = probabilities[i, 0]
                target = targets[i, 0]

                prediction = prob >= THRESHOLD

                if label == "Normal":
                    # All pixels in a Normal image.
                    normal_probs.append(prob.flatten())

                    normal_image_stats.append({
                        "max_probability": prob.max().item(),
                        "mean_probability": prob.mean().item(),
                        "positive_fraction": (
                            prediction.float().mean().item()
                        ),
                    })

                elif label in ("Benign", "Malignant"):
                    # Pixels belonging to the ground-truth lesion.
                    if target.any():
                        lesion_probs.append(prob[target])

                    # Background pixels in images containing a lesion.
                    background = ~target
                    if background.any():
                        lesion_background_probs.append(
                            prob[background]
                        )

                    intersection = (
                        prediction & target
                    ).sum().item()

                    denominator = (
                        prediction.sum().item()
                        + target.sum().item()
                    )

                    dice = (
                        2 * intersection / denominator
                        if denominator > 0
                        else 1.0
                    )

                    lesion_image_stats.append({
                        "label": label,
                        "max_probability": prob.max().item(),
                        "mean_probability": prob.mean().item(),
                        "positive_fraction": (
                            prediction.float().mean().item()
                        ),
                        "dice": dice,
                    })

                if (
                    label in examples
                    and len(examples[label]) < MAX_EXAMPLES_PER_CLASS
                ):
                    examples[label].append({
                        "image": image,
                        "target": target,
                        "probability": prob,
                        "label": label,
                    })

    if not normal_probs:
        raise RuntimeError("No Normal images were found in validation.")

    if not lesion_probs:
        raise RuntimeError(
            "No foreground lesion pixels were found in validation."
        )

    if not lesion_background_probs:
        raise RuntimeError(
            "No lesion-background pixels were found in validation."
        )

    # ---------------------------------------------------------
    # 1. Three-group probability histogram
    # ---------------------------------------------------------
    normal_values = torch.cat(normal_probs).numpy()
    lesion_values = torch.cat(lesion_probs).numpy()
    background_values = torch.cat(
        lesion_background_probs
    ).numpy()

    plt.figure(figsize=(10, 5))

    plt.hist(
        normal_values,
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="Normal image pixels",
    )

    plt.hist(
        background_values,
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="Background in lesion images",
    )

    plt.hist(
        lesion_values,
        bins=50,
        range=(0, 1),
        density=True,
        alpha=0.5,
        label="Ground-truth lesion pixels",
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label=f"Threshold = {THRESHOLD}",
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Normal, lesion-background, and lesion probabilities")
    plt.legend()
    plt.tight_layout()

    histogram_path = (
        OUTPUT_DIR / "three_group_probability_histogram.png"
    )

    plt.savefig(histogram_path, dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # 2. Example images, masks, probabilities, and predictions
    # ---------------------------------------------------------
    for label, samples in examples.items():
        if not samples:
            continue

        fig, axes = plt.subplots(
            len(samples),
            5,
            figsize=(16, 3.5 * len(samples)),
            squeeze=False,
        )

        for row, sample in enumerate(samples):
            image = sample["image"].permute(1, 2, 0).numpy()
            target = sample["target"].numpy()
            prob = sample["probability"].numpy()
            pred = prob >= THRESHOLD

            axes[row, 0].imshow(image)
            axes[row, 0].set_title("Image")

            axes[row, 1].imshow(
                target,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, 1].set_title("Ground truth")

            axes[row, 2].imshow(
                prob,
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            axes[row, 2].set_title("Probability")

            axes[row, 3].imshow(
                pred,
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, 3].set_title("Prediction")

            # Overlay: green = true positive, red = false positive,
            # blue = false negative.
            overlay = image.copy()

            if overlay.max() > 1:
                overlay = overlay / 255.0

            overlay = overlay.astype(float)

            tp = pred & target
            fp = pred & ~target
            fn = ~pred & target

            overlay[tp] = [0.0, 1.0, 0.0]
            overlay[fp] = [1.0, 0.0, 0.0]
            overlay[fn] = [0.0, 0.0, 1.0]

            axes[row, 4].imshow(overlay)
            axes[row, 4].set_title(
                "Overlay: TP green, FP red, FN blue"
            )

            for ax in axes[row]:
                ax.axis("off")

        fig.tight_layout()

        example_path = (
            OUTPUT_DIR / f"{label.lower()}_examples_detailed.png"
        )

        fig.savefig(example_path, dpi=150)
        plt.close(fig)

    # ---------------------------------------------------------
    # 3. Print probability distribution summaries
    # ---------------------------------------------------------
    def print_distribution(name, values):
        tensor = torch.as_tensor(values, dtype=torch.float32)

        print(f"\n{name}:")
        print(f"  Pixel count: {tensor.numel():,}")
        print(f"  Mean probability: {tensor.mean().item():.4f}")
        print(f"  Median probability: {tensor.median().item():.4f}")
        print(f"  90th percentile: {torch.quantile(tensor, 0.90).item():.4f}")
        print(f"  95th percentile: {torch.quantile(tensor, 0.95).item():.4f}")
        print(f"  99th percentile: {torch.quantile(tensor, 0.99).item():.4f}")
        print(
            f"  Fraction >= {THRESHOLD:.2f}: "
            f"{(tensor >= THRESHOLD).float().mean().item():.4f}"
        )

    print("=" * 60)
    print("Prediction probability analysis")
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Threshold:", THRESHOLD)
    print("Device:", device)
    print("Validation images:", len(dataset))
    print("Normal images:", len(normal_image_stats))
    print("Lesion images:", len(lesion_image_stats))

    print_distribution("Normal image pixels", normal_values)
    print_distribution(
        "Background pixels in lesion images",
        background_values,
    )
    print_distribution("Ground-truth lesion pixels", lesion_values)

    normal_positive_fractions = [
        item["positive_fraction"] for item in normal_image_stats
    ]

    normal_fp_rate = (
        sum(value > 0 for value in normal_positive_fractions)
        / max(len(normal_positive_fractions), 1)
    )

    print("\nPer-image summary:")
    print(f"  Normal image FP rate: {normal_fp_rate:.4f}")
    print(
        "  Mean Normal positive fraction: "
        f"{sum(normal_positive_fractions) / len(normal_positive_fractions):.4f}"
    )
    print(
        "  Mean lesion-image Dice: "
        f"{sum(item['dice'] for item in lesion_image_stats) / len(lesion_image_stats):.4f}"
    )

    print("\nSaved outputs:")
    print("  Histogram:", histogram_path)
    print("  Normal examples:", OUTPUT_DIR / "normal_examples_detailed.png")
    print("  Benign examples:", OUTPUT_DIR / "benign_examples_detailed.png")
    print("  Malignant examples:", OUTPUT_DIR / "malignant_examples_detailed.png")


if __name__ == "__main__":
    main()
