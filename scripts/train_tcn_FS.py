import os
import sys
import shutil
from datetime import datetime
import time

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.append(SRC)

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import wandb
import mirdata

from pre.data_loader import BeatData
from model.tcn import MultiTracker
from model.lightning_module import PLTCN
from utils.split_utils import carn_split_keys

import numpy as np
import random


# ----- Load Dataset -----
data_home = '../../../datasets/'

carn = mirdata.initialize('compmusic_carnatic_rhythm', version='full_dataset_1.0', data_home=data_home, )
carn.download(['index'])
#carn.download() # run once and comment line
#carn.validate()
carn_tracks = carn.load_tracks()
carn_keys = list(carn_tracks.keys())

# ----- Device Setup -----
if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
else:
    device = torch.device("cpu")
    accelerator = "cpu" 
    
num_workers = 0  # Set to 0 for cpu

# ----- Set Training Parameters -----
train_fold = 2
test_fold = 1

PARAMS = {
    "LEARNING_RATE": 0.005,
    "N_FILTERS": 20,
    "KERNEL_SIZE": 5,
    "DROPOUT": 0.15,
    "N_DILATIONS": 11,
    "N_EPOCHS": 100,
    "LOSS": "BCE",
    "POST_PROCESSOR": "JOINT"
}

# ----- Set seeds -----
for run, seed in enumerate([42, 52, 62], start=1):

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)

    # ----- Load Splits -----
    csv_path = os.path.join(ROOT, 'data', 'cmr_splits.csv')

    carn_train_keys, carn_val_keys, carn_test_keys = carn_split_keys(
                                                            csv_path=csv_path,
                                                            train_fold=train_fold,
                                                            test_fold=test_fold,
                                                            split_col='Taala',
                                                            test_size=0.2,
                                                            seed=seed,
                                                            reorder=True
                                                        )


    # Prepare datasets and loaders
    train_data = BeatData(carn_tracks, carn_train_keys, widen=True)
    val_data = BeatData(carn_tracks, carn_val_keys, widen=True)
    test_data = BeatData(carn_tracks, carn_test_keys, widen=True)

    train_loader = DataLoader(train_data, batch_size=1, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=num_workers)

    # ----- Model and Lightning Module -----
    tcn = MultiTracker(
        n_filters=PARAMS["N_FILTERS"],
        n_dilations=PARAMS["N_DILATIONS"],
        kernel_size=PARAMS["KERNEL_SIZE"],
        dropout_rate=PARAMS["DROPOUT"]
    )

    model = PLTCN(model=tcn, params=PARAMS)

    # ----- Callbacks -----
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    ckpt_name = f"tcn_carnatic_fs"

    CKPTS_DIR = os.path.join(ROOT, 'output', 'checkpoints', timestamp)
    os.makedirs(CKPTS_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPTS_DIR,
        filename=f"{ckpt_name}-trainfold{train_fold}-run{run}" + "-{epoch:02d} -{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,       # stop training if no improvement for 20 epochs
        mode="min",
        min_delta=1e-4,  # minimum change to qualify as an improvement
        verbose=True
    )

    # ----- Loggers -----

    run_config = PARAMS.copy()
    run_config.update({
        "FOLD": train_fold,
        "RUN_ID": run,
        "SEED": seed
    })

    wandb.login(key='40ce66ff8d453431f4c75ca162a50b37f7f0f1e1')
    wandb_run = wandb.init(
        project="TCN_Carnatic_FS_100",
        name=f"TCN_carnatic_FS_{timestamp}_trainfold{train_fold}_run{run}_seed{seed}",
        config=run_config,
    )

    wandb_logger = WandbLogger(experiment=wandb_run)
    csv_logger = CSVLogger("lightning_logs")  # this gives you metrics.csv


    # ----- Trainer -----
    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        accelerator=accelerator,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[csv_logger, wandb_logger]  # explicitly include both
    )

    # ----- Train -----

    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()


    train_duration = end_time - start_time
    # Format as HH:MM:SS
    hours, rem = divmod(train_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    print(f"Total training time: {time_str} (hh:mm:ss)")

    # Log training time to W&B
    wandb.log({"train_time_sec": train_duration})

    # ----- Copy Train Logs -----
    try:
        metrics_src = os.path.join(csv_logger.log_dir, "metrics.csv")

        # Target path: output/checkpoints/<timestamp>/metrics.csv
        metrics_dest = os.path.join(CKPTS_DIR, "metrics.csv")

        # Copy it
        shutil.copyfile(metrics_src, metrics_dest)
        print(f"Copied Lightning metrics.csv to: {metrics_dest}")
        
    finally:
        # Clean up the wandb run directory
        wandb_run.finish()
        print("Wandb run finished")
