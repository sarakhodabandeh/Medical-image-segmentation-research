import sys

from src.utils.paths import PROJECT_ROOT

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
