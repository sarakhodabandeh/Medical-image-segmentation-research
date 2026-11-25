"""Tests for the train-time augmentation (src/datasets/augment.py)."""

import torch

from src.datasets.augment import augment_batch


def batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(4, 3, 64, 64, generator=generator)
    masks = torch.zeros(4, 1, 64, 64)
    masks[:, :, 20:40, 24:44] = 1.0
    return images, masks


def test_shapes_range_and_binary_masks():
    images, masks = batch()
    out_images, out_masks = augment_batch(images, masks, torch.Generator().manual_seed(1))
    assert out_images.shape == images.shape and out_masks.shape == masks.shape
    assert out_images.min() >= 0.0 and out_images.max() <= 1.0
    assert set(out_masks.unique().tolist()) <= {0.0, 1.0}


def test_reproducible_for_the_same_seed():
    images, masks = batch()
    a = augment_batch(images, masks, torch.Generator().manual_seed(7))
    b = augment_batch(images, masks, torch.Generator().manual_seed(7))
    c = augment_batch(images, masks, torch.Generator().manual_seed(8))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_identity_settings_leave_the_batch_unchanged():
    images, masks = batch()
    identity = {
        "hflip_prob": 0.0,
        "rotation_degrees": 0.0,
        "scale_range": [1.0, 1.0],
        "brightness": 0.0,
        "contrast": 0.0,
        "gamma_range": [1.0, 1.0],
    }
    out_images, out_masks = augment_batch(images, masks, torch.Generator().manual_seed(0), identity)
    assert torch.allclose(out_images, images, atol=1e-5)
    assert torch.equal(out_masks, masks)


def test_flip_moves_image_and_mask_together():
    images, masks = batch()
    flip_only = {
        "hflip_prob": 1.0,
        "rotation_degrees": 0.0,
        "scale_range": [1.0, 1.0],
        "brightness": 0.0,
        "contrast": 0.0,
        "gamma_range": [1.0, 1.0],
    }
    out_images, out_masks = augment_batch(
        images, masks, torch.Generator().manual_seed(0), flip_only
    )
    assert torch.allclose(out_images, images.flip(-1), atol=1e-5)
    assert torch.equal(out_masks, masks.flip(-1))
