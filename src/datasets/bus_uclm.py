import csv
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.paths import PROJECT_ROOT

DEFAULT_ZIP_PATH = (
    PROJECT_ROOT / "data" / "raw" / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "bus_uclm_patient_split.csv"

ZIP_ROOT = "BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM"


class BUSUCLMDataset(Dataset):
    def __init__(
        self,
        split=None,
        zip_path=DEFAULT_ZIP_PATH,
        split_path=DEFAULT_SPLIT_PATH,
        cache_in_memory=False,
    ):
        # cache_in_memory keeps each decoded image and mask (uint8 / bool) after the first read,
        # so later epochs skip the zip and PNG decoding. Off by default (paper v1 behaviour).
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        self.zip_path = Path(zip_path)
        self.split_path = Path(split_path)

        if not self.zip_path.exists():
            raise FileNotFoundError(f"Dataset ZIP not found: {self.zip_path}")

        if not self.split_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_path}")

        allowed_splits = {"train", "val", "test"}

        if split is not None and split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{split}'. Choose one of {sorted(allowed_splits)} or None."
            )

        # Read the saved patient-level split.
        with self.split_path.open("r", newline="", encoding="utf-8-sig") as file:
            split_rows = list(csv.DictReader(file))

        split_by_filename = {row["filename"]: row["split"] for row in split_rows}

        if len(split_by_filename) != len(split_rows):
            raise ValueError("Duplicate filenames found in the split CSV.")

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

                if image_path not in archive_names or mask_path not in archive_names:
                    raise FileNotFoundError(f"Missing image or mask for {filename}")

                with Image.open(BytesIO(archive.read(image_path))) as image:
                    image_size = image.size

                with Image.open(BytesIO(archive.read(mask_path))) as mask:
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
            raise RuntimeError(f"No valid samples found for split={split!r}.")

        # Verify that every sample belongs to the requested split.
        if split is not None:
            assert all(sample["split"] == split for sample in self.samples)

    def _read(self, sample):
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            with Image.open(BytesIO(archive.read(sample["image_path"]))) as image:
                image_uint8 = np.array(image.convert("RGB"), dtype=np.uint8)

            with Image.open(BytesIO(archive.read(sample["mask_path"]))) as mask:
                mask_rgb = np.array(mask.convert("RGB"))

        foreground = np.any(mask_rgb != 0, axis=2)
        return image_uint8, foreground

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.cache_in_memory and index in self._cache:
            image_uint8, foreground = self._cache[index]
        else:
            image_uint8, foreground = self._read(sample)
            if self.cache_in_memory:
                self._cache[index] = (image_uint8, foreground)

        # Convert image to a float tensor in [0, 1].
        image = torch.from_numpy(image_uint8.transpose(2, 0, 1).astype(np.float32) / 255.0)

        # Binary mask: background=0, lesion=1.
        mask = torch.from_numpy(foreground.astype(np.float32)[None, :, :])

        return {
            "image": image,
            "mask": mask,
            "filename": sample["filename"],
            "patient_id": sample["patient_id"],
            "label": sample["label"],
            "split": sample["split"],
        }
