import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


TARGET_FILENAMES = [
    "CAWI_005.png",
    "CAWI_010.png",
    "CODE_000.png",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "test_evaluation"
    / "original_data_check"
)
TARGET_FILENAMES = [
    "CAWI_005.png",
    "CAWI_010.png",
    "CODE_000.png",
]

OUTPUT_DIR = ROOT / "results" / "test_evaluation" / "original_data_check"


def main():
    dataset = BUSUCLMDataset(split="test")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = {
        sample["filename"]: sample
        for sample in dataset.samples
        if sample["filename"] in TARGET_FILENAMES
    }

    missing = set(TARGET_FILENAMES) - set(selected)
    if missing:
        print("Missing samples:", sorted(missing))

    import zipfile

    with zipfile.ZipFile(dataset.zip_path, "r") as archive:
        for filename in TARGET_FILENAMES:
            if filename not in selected:
                continue

            sample = selected[filename]

            with Image.open(
                archive.open(sample["image_path"])
            ) as img:
                image = np.array(img.convert("RGB"))

            with Image.open(
                archive.open(sample["mask_path"])
            ) as msk:
                mask_rgb = np.array(msk.convert("RGB"))

            mask = np.any(mask_rgb != 0, axis=2)

            print(f"\n{filename}")
            print("Label:", sample["label"])
            print("Original image shape:", image.shape)
            print("Original mask shape:", mask_rgb.shape)
            print("Mask foreground pixels:", int(mask.sum()))
            print("Unique mask RGB colors:",
                  np.unique(mask_rgb.reshape(-1, 3), axis=0)[:10])

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(image)
            axes[0].set_title("Original image")

            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Original binary mask")

            axes[2].imshow(image)
            axes[2].imshow(mask, cmap="autumn", alpha=0.45)
            axes[2].set_title("Mask over original image")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            output_path = OUTPUT_DIR / f"{Path(filename).stem}_original.png"
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
            plt.close(fig)

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
