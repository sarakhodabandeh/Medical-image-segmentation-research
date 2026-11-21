"""Tests for the v2 evaluation helpers (no dataset or checkpoint needed)."""

import importlib.util

import numpy as np

from src.utils.paths import PROJECT_ROOT

spec = importlib.util.spec_from_file_location(
    "evaluate_v2", PROJECT_ROOT / "scripts" / "evaluation" / "evaluate_v2.py"
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_valid_region_matches_resize_with_padding_geometry():
    region = ev.valid_region(96, 128, 256)  # scaled to 192 x 256, 32 rows of padding each side
    assert region.sum() == 192 * 256
    assert not region[:32].any() and not region[-32:].any() and region[32:224].all()


def test_remove_small_drops_only_small_components():
    prediction = np.zeros((40, 40), dtype=bool)
    prediction[2:4, 2:4] = True  # 4 pixels
    prediction[10:20, 10:20] = True  # 100 pixels
    kept = ev.remove_small(prediction, 10)
    assert kept[15, 15] and not kept[3, 3]
    assert kept.sum() == 100


def test_detection_counts_hits_and_false_components():
    target = np.zeros((60, 60), dtype=bool)
    target[10:20, 10:20] = True  # one lesion
    prediction = np.zeros_like(target)
    prediction[12:22, 12:22] = True  # overlaps the lesion (IoU 0.47)
    prediction[40:50, 40:50] = True  # separate false-positive component
    n_gt, detected, false_components = ev.detection(prediction, target)
    assert (n_gt, detected, false_components) == (1, 1, 1)


def test_image_metrics_for_normal_and_lesion_images():
    region = np.ones((32, 32), dtype=bool)
    region[:, :4] = False
    probability = np.zeros((32, 32))
    probability[0:12, 0:12] = 0.9  # 144 pixels, 48 of them in the padding
    empty = np.zeros((32, 32), dtype=bool)

    normal = ev.image_metrics(probability, empty, region, 0.5, "Normal")
    assert normal["fp_image_valid"] == 1 and normal["fp_image_full"] == 1
    assert np.isclose(normal["fp_area_fraction_valid"], 96 / region.sum())
    assert normal["fp_image_valid_min100"] == 0  # the valid-region part is 96 pixels

    target = np.zeros((32, 32), dtype=bool)
    target[0:12, 4:12] = True
    lesion = ev.image_metrics(probability, target, region, 0.5, "Benign")
    assert np.isclose(lesion["dice_valid"], 1.0)
    assert lesion["lesion_sensitivity_valid"] == 1.0
