from pathlib import Path

import yaml


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
CONFIG_PATH = PROJECT_ROOT / "configs" / "unet.yaml"


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["experiment"]["seed"] == 42
    assert config["data"]["image_size"] == 256
    assert config["data"]["batch_size"] == 4
    assert config["training"]["epochs"] == 50
    assert config["loss"]["bce_weight"] + config["loss"]["dice_weight"] == 1.0

    print("Configuration loaded successfully.")
    print("Experiment:", config["experiment"]["name"])
    print("Image size:", config["data"]["image_size"])
    print("Batch size:", config["data"]["batch_size"])
    print("Epochs:", config["training"]["epochs"])
    print("Learning rate:", config["training"]["learning_rate"])
    print("Configuration test PASSED!")


if __name__ == "__main__":
    main()