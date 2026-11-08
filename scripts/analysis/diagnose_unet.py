diagnose_unet.py
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]
    size = config["data"]["image_size"]

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Saved best validation Dice:",
          checkpoint["best_val_lesion_dice"])
    print("Preprocessing: resize with padding")
    print("Thresholds:", THRESHOLDS)

    for split in ["train", "val"]:
        dataset = BUSUCLMDataset(split=split)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        records = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                images, masks = resize_with_padding(
                    images, masks, size
                )

                probabilities = torch.sigmoid(model(images))

                for i, label in enumerate(batch["label"]):
                    records.append({
                        "filename": batch["filename"][i],
                        "label": label,
                        "probability": probabilities[i, 0].cpu(),
                        "target": masks[i, 0].cpu() >= 0.5,
                    })

        print(f"\n--- {split.upper()} ---")

        # Inspect probability distributions for the first 8 samples.
        for record in records[:8]:
            prob = record["probability"]
            target = record["target"]

            print(
                f"{record['filename']} | {record['label']} | "
                f"true_fraction={target.float().mean().item():.4f} | "
                f"prob_mean={prob.mean().item():.4f} | "
                f"prob_min={prob.min().item():.4f} | "
                f"prob_max={prob.max().item():.4f}"
            )

        print("\nThreshold analysis:")
        print(
            "Threshold | Lesion Dice | Normal FP rate | "
            "Mean normal positive fraction"
        )

        for threshold in THRESHOLDS:
            lesion_dices = []
            normal_fp = 0
            normal_count = 0
            normal_fractions = []

            for record in records:
                pred = record["probability"] >= threshold
                target = record["target"]
                label = record["label"]

                if label in ("Benign", "Malignant"):
                    intersection = (pred & target).sum().item()
                    denominator = (
                        pred.sum().item() + target.sum().item()
                    )

                    dice = (
                        2 * intersection / denominator
                        if denominator > 0
                        else 1.0
                    )
                    lesion_dices.append(dice)

                elif label == "Normal":
                    normal_count += 1
                    fraction = pred.float().mean().item()
                    normal_fractions.append(fraction)

                    if pred.any().item():
                        normal_fp += 1

            mean_dice = (
                sum(lesion_dices) / len(lesion_dices)
                if lesion_dices else 0.0
            )
            fp_rate = normal_fp / max(normal_count, 1)
            mean_fraction = (
                sum(normal_fractions) / len(normal_fractions)
                if normal_fractions else 0.0
            )

            print(
                f"{threshold:9.2f} | {mean_dice:11.4f} | "
                f"{fp_rate:14.4f} | {mean_fraction:29.4f}"
            )


if __name__ == "__main__":
    main()
