import sys
from pathlib import Path

import torch


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


from src.models.unet import UNet

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ).to(device)

    # Use a small test input first to check the architecture.
    x = torch.randn(1, 3, 256, 256, device=device)

    with torch.no_grad():
        output = model(x)

    total_params = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Device:", device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Parameters: {total_params:,}")

    assert output.shape == (1, 1, 256, 256)
    print("U-Net test PASSED!")


if __name__ == "__main__":
    main()
