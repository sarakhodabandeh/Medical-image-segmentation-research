
import csv
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bus_uclm_patient_split.csv"
)

ZIP_ROOT = (
    "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"
)


class BUSUCLMDataset(Dataset):
    def __init__(
        self,
        split=None,
        zip_path=DEFAULT_ZIP_PATH,
        split_path=DEFAULT_SPLIT_PATH,
    ):
        self.zip_path = Path(zip_path)
        self.split_path = Path(split_path)

        if not self.zip_path.exists():
            raise FileNotFoundError(
                f"Dataset ZIP not found: {self.zip_path}"
            )

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_path}"
            )

        allowed_splits = {"train", "val", "test"}

        if split is not None and split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Choose one of {sorted(allowed_splits)} or None."
            )

        # Read the saved patient-level split.
        with self.split_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            split_rows = list(csv.DictReader(file))

        split_by_filename = {
            row["filename"]: row["split"]
            for row in split_rows
        }

        if len(split_by_filename) != len(split_rows):
            raise ValueError(
                "Duplicate filenames found in the split CSV."
            )

        self.samples = []

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            archive_names = set(archive.namelist())
            info_path = f"{ZIP_ROOT}/INFO.csv"

            with archive.open(info_path, "r") as file:
                reader = csv.DictReader(
                    (line.decode("utf-8-sig") for line in file),
                    delimiter=";",
                )
                metadata = list(reader)

            for row in metadata:
                filename = row["Image"]
                image_path = f"{ZIP_ROOT}/images/{filename}"
                mask_path = f"{ZIP_ROOT}/masks/{filename}"

                if filename not in split_by_filename:
                    continue

                assigned_split = split_by_filename[filename]

                if split is not None and assigned_split != split:
                    continue

                if (
                    image_path not in archive_names
                    or mask_path not in archive_names
                ):
                    raise FileNotFoundError(
                        f"Missing image or mask for {filename}"
                    )

                with Image.open(
                    BytesIO(archive.read(image_path))
                ) as image:
                    image_size = image.size

                with Image.open(
                    BytesIO(archive.read(mask_path))
                ) as mask:
                    mask_size = mask.size

                # Exclude unresolved image-mask dimension mismatches.
                if image_size != mask_size:
                    continue

                self.samples.append(
                    {
                        "filename": filename,
                        "patient_id": filename.split("_")[0],
                        "label": row["Label"],
                        "split": assigned_split,
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found for split={split!r}."
            )

        # Verify that every sample belongs to the requested split.
        if split is not None:
            assert all(
                sample["split"] == split
                for sample in self.samples
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            with Image.open(
                BytesIO(archive.read(sample["image_path"]))
            ) as image:
                image = np.array(
                    image.convert("RGB"),
                    dtype=np.float32,
                )

            with Image.open(
                BytesIO(archive.read(sample["mask_path"]))
            ) as mask:
                mask_rgb = np.array(mask.convert("RGB"))

        # Convert image to a float tensor in [0, 1].
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy() / 255.0
        )

        # Binary mask: background=0, lesion=1.
        foreground = np.any(mask_rgb != 0, axis=2)
        mask = torch.from_numpy(
            foreground.astype(np.float32)[None, :, :]
        )

        return {
            "image": image,
            "mask": mask,
            "filename": sample["filename"],
            "patient_id": sample["patient_id"],
            "label": sample["label"],
            "split": sample["split"],
        }
