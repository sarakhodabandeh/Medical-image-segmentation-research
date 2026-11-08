
import torch
import torch.nn.functional as F


def resize_with_padding(images, masks, size=256):
    """
    Resize images while preserving aspect ratio, then pad to size x size.

    images: [B, 3, H, W], float tensor in [0, 1]
    masks:  [B, 1, H, W], binary float tensor in [0, 1]

    Returns:
        resized_images: [B, 3, size, size]
        resized_masks:  [B, 1, size, size]
    """
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("Images and masks must be 4D tensors.")

    if images.shape[0] != masks.shape[0]:
        raise ValueError("Image and mask batch sizes must match.")

    if images.shape[-2:] != masks.shape[-2:]:
        raise ValueError("Image and mask dimensions must match.")

    _, _, height, width = images.shape

    scale = min(size / height, size / width)

    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    resized_images = F.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

    resized_masks = F.interpolate(
        masks,
        size=(new_height, new_width),
        mode="nearest",
    )

    pad_height = size - new_height
    pad_width = size - new_width

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Image padding: replicate border pixels to avoid artificial black borders.
    resized_images = F.pad(
        resized_images,
        (left, right, top, bottom),
        mode="replicate",
    )

    # Mask padding must remain background.
    resized_masks = F.pad(
        resized_masks,
        (left, right, top, bottom),
        mode="constant",
        value=0,
    )

    return resized_images, resized_masks

def resize_with_constant_padding(images, masks, size=256):
    """
    Preserve aspect ratio, then pad images with a constant value.
    Mask padding remains background (0).
    """
    if images.ndim != 4 or masks.ndim != 4:
        raise ValueError("Images and masks must be 4D tensors.")

    if images.shape[-2:] != masks.shape[-2:]:
        raise ValueError("Image and mask dimensions must match.")

    _, _, height, width = images.shape
    scale = min(size / height, size / width)

    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    images = torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

    masks = torch.nn.functional.interpolate(
        masks,
        size=(new_height, new_width),
        mode="nearest",
    )

    pad_height = size - new_height
    pad_width = size - new_width

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    images = torch.nn.functional.pad(
        images,
        (left, right, top, bottom),
        mode="constant",
        value=0.0,
    )

    masks = torch.nn.functional.pad(
        masks,
        (left, right, top, bottom),
        mode="constant",
        value=0.0,
    )

    return images, masks
