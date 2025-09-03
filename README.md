# TCN Carnatic Tracker

This repository is a companion to the Master's thesis "Revisiting Meter Tracking in Carnatic Music using Deep Learning Approaches," submitted towards a Master in Sound and Music Computing at Universitat Pompeu Fabra (August 2025). It includes an implementation of Temporal Convolutional Network (TCN) for beat and downbeat tracking, with baselines (TCN_BL), from-scratch (TCN-FS) and fine-tuning (TCN-FT) setups for the CompMusic Carnatic Music Rhythm Dataset (CMR). The code can be used to reproduce the study's results; pretrained checkpoints and their evaluation reported in the study are provided under `pretrained/` and `output/results/` respectively.

## Contents
- [Installation](#installation)
  - [Patching madmom](#apply-madmom-patch)
  - [Weights & Biases (wandb)](#weights--biases-wandb)
- [Dataset](#dataset)
- [Training](#training)
  - [Baseline (TCN-BL) on multiple Western datasets](#baseline-on-multiple-western-datasets-tcn-bl)
  - [From Scratch (TCN-FS) on CMR](#from-scratch-on-cmr-tcn-fs)
  - [Fine-Tune (TCN-FT) on CMR](#fine-tune-on-cmr-tcn-ft)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Notebooks](#notebooks)

## Installation

Clone this repository and navigate into it:

```bash
git clone <repo_url>
cd tcn-carnatic-tracker
```

Optionally, install in a Conda environment from `requirements.txt`.

```bash
conda create -n tcn-carn python=3.10 -y
conda activate tcn-carn
pip install -r requirements.txt
```

### Apply madmom patch
After installing dependencies, patch [madmom](https://github.com/CPJKU/madmom) once to fix `MutableSequence` import:

```bash
python madmom_patch.py
```

### Weights & Biases (wandb)
- This project supports online logging using wandb
- Scripts default to online logging unless `--disable-wandb` is passed.
- Config files also include a `wandb_api_key` field required by scripts.

To enable online logging:
1. Create a free account at https://wandb.ai
2. Find your API key in Settings → API key
3. Update config files in `config/` with your API key
4. Run training normally. Use `--disable-wandb` to turn logging off.

## Dataset

- Primary dataset: [CompMusic Carnatic Rhythm Dataset (CMR)](https://zenodo.org/records/1264394#.WyeLDByxXMU).
    - Expected structure: place the entire `CMR_full_dataset_1.0` folder inside `data/` or your preferred folder (to be passed as `data_home` in the scripts).

- Data loading, validation and parsing via [`mirdata`](https://mirdata.readthedocs.io/en/stable/). We use a custom fork with pre-saved and edited index files. 

## Training

Configure model/training in `config/model.yaml` and `config/train_{BL,FS,FT}.yaml`.

All training scripts live in `scripts/`. They read hyperparameters from YAML configs and log to Lightning CSV and wandb. Checkpoints are saved to timestamped folders under `output/checkpoints/` and automatically copied to `pretrained/`.

### Baseline on multiple Western datasets (TCN-BL)
Script: `scripts/train_BL.py`

Baseline (TCN-BL) configured and trained on: `gtzan_genre`, `beatles`, `ballroom`, `rwc_popular`, `rwc_jazz`, `rwc_classical`

- Arguments:
  - `--data_home`: root folder containing datasets (configured for use with mirdata).
  - `--datasets`: list of dataset names (optional). Set to all by default. Can be trained on one or more of the above datasets.
  - `--disable-wandb`: turn off wandb.
- Uses `config/train_BL.yaml` + `config/model.yaml`.
- Runs 3 seeds `[42, 52, 62]`, creates stratified splits, trains, and saves best checkpoints to `pretrained/tcn_bl/` as `tcn_bl-run{run}.ckpt`.

Example:
```bash
cd scripts
python train_BL.py --data_home ../data --datasets gtzan_genre beatles ballroom rwc_popular rwc_jazz rwc_classical
```

### From Scratch on CMR (TCN-FS)
Script: `scripts/train_FS.py`

- Trains only on CMR with 2-fold train/test split.
- Predetermined train/test splits read from `data/cmr_splits.csv`
- `train_fold` in {1,2}; `test_fold` is the other.
- Arguments:
  - `--data_home`: root containing `CMR_full_dataset_1.0`.
  - `--disable-wandb`: turn off wandb.
- Uses `config/train_FS.yaml` + `config/model.yaml`.
- For each `train_fold` in `[1,2]` and seeds `[42, 52, 62]`, trains and writes checkpoints to `pretrained/tcn_carnatic_fs/` as `tcn_carnatic_fs-trainfold{fold}-run{run}.ckpt`.

Example:
```bash
cd scripts
python train_FS.py --data_home ../data
```

### Fine-Tune on CMR (TCN-FT)
Script: `scripts/train_FT.py`

- Fine-tunes on CMR using a pretrained checkpoint specified in `config/train_FT.yaml` via `training.pretrained_model`.
- Arguments:
  - `--data_home`: root containing `CMR_full_dataset_1.0`.
  - `--disable-wandb`: turn off wandb.
- Uses `config/train_FT.yaml` + `config/model.yaml`.
- For `train_fold` in `[2]` and seeds `[42, 52, 62]`, trains and writes checkpoints to `pretrained/tcn_carnatic_ft/` as `tcn_carnatic_ft-trainfold{fold}-run{run}.ckpt`.

Example:
```bash
cd scripts
python train_FT.py --data_home ../data
```

## Evaluation

Script: `scripts/eval_tcn.py`

- Modes: `--mode {BL,FT,FS}`. Loads config to get model params and selects checkpoint directory under `pretrained/<ckpt_name>/`.
- For BL, evaluates on complete dataset. For FT/FS, evaluates per split using `data/cmr_splits.csv`.
- Writes CSVs to `output/results/<folder>/` matching checkpoint names.

Examples:
```bash
cd scripts
python eval_tcn.py --mode BL --data_home ../data
python eval_tcn.py --mode FS --data_home ../data
python eval_tcn.py --mode FT --data_home ../data
```

## Inference

For a detailed example with audio files, see `notebooks/carnatic_inference.ipynb`.

**Summary usage:**
- Module: `src/inference/tracker.py`
- Class `Audio2Beats(checkpoint_path, model_params=None, post_processor="JOINT")`.
- Example:
```python
from src.inference.tracker import Audio2Beats

model = Audio2Beats(checkpoint_path="pretrained/tcn_carnatic_ft/tcn_carnatic_ft-trainfold2-run1.ckpt")

pred = model.predict("/path/to/audio.wav")
# pred contains beat/downbeat times from the post-processor
```

## Notebooks

The notebooks provided contain the code to generate all the analysis and plots presented in the thesis.

- `notebooks/dataset_analysis.ipynb`: Dataset statistics and inspection.
- `notebooks/outlier_analysis.ipynb`: Outlier exploration.
- `notebooks/tempo_analysis.ipynb`: Tempo-focused analysis.
- `notebooks/carnatic_inference.ipynb`: Example inference on audio.