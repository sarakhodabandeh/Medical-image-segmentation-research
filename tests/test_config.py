"""Checks that configs/paper_unet.yaml holds the settings reported in the paper."""

import pytest
import yaml

from src.utils.paths import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "paper_unet.yaml"


@pytest.fixture(scope="module")
def config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_data_settings(config):
    assert config["data"]["image_size"] == 256
    assert config["data"]["batch_size"] == 4


def test_model_settings(config):
    assert config["model"]["in_channels"] == 3
    assert config["model"]["out_channels"] == 1
    assert config["model"]["base_channels"] == 16


def test_training_settings(config):
    training = config["training"]
    assert training["epochs"] == 10
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0001
    assert training["early_stopping_patience"] == 10
    assert training["use_amp"] is True


def test_loss_settings(config):
    loss = config["loss"]
    assert loss["bce_weight"] + loss["dice_weight"] == 1.0
    assert loss["pos_weight"] == 5.0


def test_config_contains_no_file_paths(config):
    # The split CSV and the output folder are command-line arguments of the scripts.
    assert "split_csv" not in config.get("data", {})
    assert "output" not in config
