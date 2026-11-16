import sys
from pathlib import Path

import torch


from src.utils.paths import PROJECT_ROOT

from src.models.attention_unet import AttentionUNet

def test_attention_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionUNet(
        in_channels=3,
        out_channels=1,
        base_channels=16,
    ).to(device)

    x = torch.randn(4, 3, 256, 256, device=device)

    with torch.no_grad():
        output = model(x)

    print("Device:", device)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    assert output.shape == (4, 1, 256, 256)
    assert torch.isfinite(output).all()

    print("Attention U-Net test passed!")
