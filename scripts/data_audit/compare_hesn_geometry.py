
import zipfile
from pathlib import Path
from io import BytesIO
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

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip"
)

ROOT = (
    "BUS-UCLM Breast ultrasound lesion segmentation dataset/"
    "BUS-UCLM"
)

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    for i in range(13):
        filename = f"HESN_{i:03d}.png"

        image_path = f"{ROOT}/images/{filename}"
        mask_path = f"{ROOT}/masks/{filename}"

        with Image.open(BytesIO(archive.read(image_path))) as im:
            image_width, image_height = im.size

        with Image.open(BytesIO(archive.read(mask_path))) as im:
            mask_width, mask_height = im.size

        image_ratio = image_width / image_height
        mask_ratio = mask_width / mask_height

        print(
            f"{filename}: "
            f"image={image_width}x{image_height}, "
            f"mask={mask_width}x{mask_height}, "
            f"image_ratio={image_ratio:.4f}, "
            f"mask_ratio={mask_ratio:.4f}"
        )
