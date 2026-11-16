import sys
from pathlib import Path

import torch


from src.utils.paths import PROJECT_ROOT

from src.models.unet import UNet

def test_unet():
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
