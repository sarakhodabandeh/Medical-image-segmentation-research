
from pathlib import Path
import sys

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
        "Could not locate project root."
    )


PROJECT_ROOT = find_project_root()

print("=" * 60)
print("PROJECT PATH CHECK")
print("=" * 60)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

paths = [
    "README.md",
    "requirements.txt",
    "configs/unet.yaml",
    "src/datasets/bus_uclm.py",
    "src/datasets/preprocessing.py",
    "src/models/unet.py",
    "src/models/attention_unet.py",
    "data",
    "results",
    ".venv",
]

print("\nChecking important files and folders:")

for item in paths:
    path = PROJECT_ROOT / item
    status = "FOUND" if path.exists() else "NOT FOUND"
    print(f"{status:12} | {item}")

print("\n" + "=" * 60)
print("PATH CHECK COMPLETED")
print("=" * 60)
