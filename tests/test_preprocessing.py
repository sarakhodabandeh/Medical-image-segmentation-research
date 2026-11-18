"""Checks for resize_with_padding: output size, binary masks, aspect ratio and padding."""

import torch

from src.datasets.preprocessing import resize_with_padding

SIZE = 256
# A 606 x 856 image is scaled by 256 / 856 to about 181 x 256, so there are
# (256 - 181) // 2 = 37 padded rows at the top and at the bottom.
MIN_PAD_ROWS = 37


def make_inputs():
    images = torch.rand(2, 3, 606, 856)
    masks = torch.zeros(2, 1, 606, 856)
    masks[0, 0, 200:400, 300:500] = 1  # a 200 x 200 square lesion in the first image
    return images, masks


def run():
    images, masks = make_inputs()
    return resize_with_padding(images, masks, size=SIZE)


def lesion_box(mask):
    rows, cols = torch.where(mask > 0)
    return rows.min().item(), rows.max().item(), cols.min().item(), cols.max().item()


def test_output_shape_and_value_range():
    out_images, out_masks = run()

    assert out_images.shape == (2, 3, SIZE, SIZE)
    assert out_masks.shape == (2, 1, SIZE, SIZE)
    assert out_images.min() >= 0.0
    assert out_images.max() <= 1.0


def test_masks_stay_binary():
    _, out_masks = run()

    assert set(torch.unique(out_masks).tolist()) <= {0.0, 1.0}
    assert out_masks[0].sum() > 0  # the lesion is not lost
    assert out_masks[1].sum() == 0  # an empty mask stays empty


def test_aspect_ratio_is_preserved():
    _, out_masks = run()
    top, bottom, left, right = lesion_box(out_masks[0, 0])

    height = bottom - top + 1
    width = right - left + 1
    assert abs(height - width) <= 2  # a square lesion stays square
    assert height > 40


def test_padding_rows_are_empty_and_lesion_is_inside_valid_region():
    _, out_masks = run()

    assert out_masks[:, :, :MIN_PAD_ROWS].sum() == 0
    assert out_masks[:, :, -MIN_PAD_ROWS:].sum() == 0

    top, bottom, _, _ = lesion_box(out_masks[0, 0])
    assert top >= MIN_PAD_ROWS
    assert bottom < SIZE - MIN_PAD_ROWS
