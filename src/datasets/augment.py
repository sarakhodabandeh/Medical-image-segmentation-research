"""Training-time data augmentation for image and mask batches.

The same random geometric transform is applied to each image and its mask; intensity changes are
applied to the image only. All randomness comes from the torch.Generator passed in, so a run is
reproducible for a given training seed. Augmentation is used for training batches only, never for
validation or test.
"""

import math

import torch
import torch.nn.functional as F

DEFAULTS = {
    "hflip_prob": 0.5,
    "rotation_degrees": 10.0,
    "scale_range": [0.9, 1.1],
    "brightness": 0.2,
    "contrast": 0.2,
    "gamma_range": [0.8, 1.2],
}


def _uniform(low, high, size, generator, device):
    values = torch.rand(size, generator=generator, device="cpu")
    return (low + (high - low) * values).to(device)


def augment_batch(images, masks, generator, settings=None):
    """Augment a batch.

    images: [B, 3, H, W] float tensor in [0, 1]
    masks:  [B, 1, H, W] binary float tensor
    generator: CPU torch.Generator that controls all random choices
    settings: dict overriding DEFAULTS

    Returns augmented (images, masks) with the same shapes; masks stay binary.
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update({k: v for k, v in settings.items() if k in DEFAULTS})

    batch, _, _, _ = images.shape
    device = images.device

    # Geometric transform: horizontal flip, rotation and isotropic scaling about the centre.
    flip = _uniform(0.0, 1.0, batch, generator, device) < cfg["hflip_prob"]
    angle = _uniform(-cfg["rotation_degrees"], cfg["rotation_degrees"], batch, generator, device)
    scale = _uniform(cfg["scale_range"][0], cfg["scale_range"][1], batch, generator, device)

    radians = angle * math.pi / 180.0
    cos, sin = torch.cos(radians) / scale, torch.sin(radians) / scale
    sign = torch.where(flip, -1.0, 1.0).to(device)

    theta = torch.zeros(batch, 2, 3, device=device, dtype=images.dtype)
    theta[:, 0, 0] = cos * sign
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin * sign
    theta[:, 1, 1] = cos

    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    images = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    masks = F.grid_sample(
        masks.to(images.dtype), grid, mode="nearest", padding_mode="zeros", align_corners=False
    )

    # Intensity transform (image only): brightness, contrast and gamma.
    shape = (batch, 1, 1, 1)
    brightness = _uniform(-cfg["brightness"], cfg["brightness"], batch, generator, device)
    contrast = _uniform(1 - cfg["contrast"], 1 + cfg["contrast"], batch, generator, device)
    gamma = _uniform(cfg["gamma_range"][0], cfg["gamma_range"][1], batch, generator, device)

    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * contrast.view(shape) + mean + brightness.view(shape)
    images = images.clamp(0.0, 1.0).pow(gamma.view(shape))

    return images.clamp(0.0, 1.0), (masks >= 0.5).to(masks.dtype)
