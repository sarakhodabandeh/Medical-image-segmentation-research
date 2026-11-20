"""Shape and sanity checks for the U-Net used in the paper."""

import pytest
import torch

from src.models.unet import UNet


@pytest.mark.parametrize("batch_size", [1, 4])
def test_unet_output_shape(batch_size):
    model = UNet(in_channels=3, out_channels=1, base_channels=16).eval()
    x = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (batch_size, 1, 256, 256)
    assert torch.isfinite(output).all()
