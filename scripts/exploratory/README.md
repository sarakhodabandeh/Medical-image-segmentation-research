# Exploratory scripts

Side analyses and earlier experiments. They are not needed to reproduce the paper.

All of them write their outputs under `results/exploratory/` (results) and
`data/processed/exploratory/` (split files), and read their configuration from
`configs/exploratory/unet.yaml`. New outputs in `results/exploratory/` are not committed.
Scripts that load an earlier checkpoint (for example `results/exploratory/unet_replicate_pw5_fp30/`)
need that checkpoint locally; checkpoints are never committed.
