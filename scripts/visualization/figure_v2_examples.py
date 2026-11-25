"""Figure: example test images with patient-level and image-level predictions (v2 runs).

Rows: the two normal test images with the largest false-positive area difference (patient-level
minus image-level) and the two lesion test images with the largest Dice difference (image-level
minus patient-level), training seed 0, threshold 0.5. The rows are chosen to illustrate the
effects, not as typical cases, and the caption should say so. Columns: input image, reference
mask, patient-level prediction, image-level prediction; Dice or false-positive area fraction
(valid region) is written under each prediction.
Needs the dataset zip and the v2 checkpoints (run after run_v2.py).
Output: results/v2/figures/examples_v2.pdf and .png.
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.datasets.bus_uclm import BUSUCLMDataset  # noqa: E402
from src.datasets.preprocessing import resize_with_padding  # noqa: E402
from src.utils.paths import PROJECT_ROOT  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "evaluate_v2", PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_v2.py"
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_examples(per_image, seed, n_each):
    rows = per_image[
        (per_image["threshold_name"] == "fixed_0.5") & (per_image["train_seed"] == seed)
    ]
    keys = ["fold", "filename", "label"]
    normal = rows[rows["label"] == "Normal"].pivot_table(
        index=keys, columns="arm", values="fp_area_fraction_valid"
    )
    normal = (normal["patient"] - normal["image"]).sort_values(ascending=False).head(n_each)
    lesion = rows[rows["label"] != "Normal"].pivot_table(
        index=keys, columns="arm", values="dice_valid"
    )
    lesion = (lesion["image"] - lesion["patient"]).sort_values(ascending=False).head(n_each)
    return list(normal.index) + list(lesion.index)


def predict_one(model, config, dataset, filename, device):
    index = next(i for i, s in enumerate(dataset.samples) if s["filename"] == filename)
    sample = dataset[index]
    image, mask = sample["image"][None].to(device), sample["mask"][None].to(device)
    h, w = image.shape[-2:]
    image, mask = resize_with_padding(image, mask, config["data"]["image_size"])
    with torch.no_grad():
        probability = torch.sigmoid(model(image).float())[0, 0].cpu().numpy()
    region = ev.valid_region(h, w, config["data"]["image_size"])
    return (
        image[0].permute(1, 2, 0).cpu().numpy(),
        mask[0, 0].cpu().numpy() >= 0.5,
        probability,
        region,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-image", default="results/v2/evaluation/test_per_image_v2.csv")
    parser.add_argument("--results-dir", default="results/v2/unet")
    parser.add_argument("--splits-dir", default="data/processed/kfold_splits")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--n-each", type=int, default=2)
    parser.add_argument("--output-dir", default="results/v2/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_image = pd.read_csv(resolve_path(args.per_image))
    examples = choose_examples(per_image, args.train_seed, args.n_each)

    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(
        len(examples), 4, figsize=(6.0, 1.55 * len(examples)), constrained_layout=True
    )
    titles = ["Input", "Reference", "Patient-level", "Image-level"]
    for row, (fold, filename, label) in enumerate(examples):
        panels = []
        for arm in ("patient", "image"):
            run_dir = (
                resolve_path(args.results_dir)
                / arm
                / f"fold{fold}"
                / f"train_seed{args.train_seed}"
            )
            model, config, _ = ev.load_model(run_dir / "best_model.pt", device)
            dataset = BUSUCLMDataset(
                split="test", split_path=resolve_path(args.splits_dir) / f"fold{fold}_{arm}.csv"
            )
            image, target, probability, region = predict_one(
                model, config, dataset, filename, device
            )
            prediction = (probability >= 0.5) & region
            if label == "Normal":
                text = f"FP area {prediction.sum() / region.sum():.3f}"
            else:
                tp = (prediction & target).sum()
                text = f"Dice {2 * tp / max(prediction.sum() + target.sum(), 1):.2f}"
            panels.append((prediction, text))
        crop = np.ix_(region.any(axis=1), region.any(axis=0))
        axes[row, 0].imshow(image[crop])
        axes[row, 1].imshow(target[crop], cmap="gray", vmin=0, vmax=1)
        for column, (prediction, text) in zip((2, 3), panels):
            axes[row, column].imshow(prediction[crop], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_xlabel(text)
        axes[row, 0].set_ylabel(f"{label}", fontsize=7)
        for column in range(4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=8)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"examples_v2.{suffix}", dpi=300)
    print(f"Saved {output_dir / 'examples_v2.pdf'} ({len(examples)} rows)")


if __name__ == "__main__":
    main()
