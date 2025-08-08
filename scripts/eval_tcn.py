import os
import sys

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.append(SRC)

import lightning as L
import torch
from torch.utils.data import DataLoader

import mirdata
import pandas as pd
import numpy as np
import random

from pre.data_loader import BeatData
from model.tcn import MultiTracker
from model.lightning_module import PLTCN
from utils.split_utils import carn_split_keys



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
#train_fold = 2
#test_fold = 1

folds = [1,2]

for fold in folds:
    train_fold = fold
    test_fold = 3 - fold  # If train_fold is 1, test_fold will be 2, and vice versa
    
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

        # ----- Load Checkpoint -----

        # from scratch
        ckpt_name = f"tcn_carnatic_fs-trainfold{train_fold}-run{run}.ckpt"
        ckpt_path = os.path.join(ROOT, 'pretrained', 'carnatic_fs_100', ckpt_name)
        
        # finetuned 
        #ckpt_name = f"tcn_carnatic_ft-trainfold{train_fold}-run{run}.ckpt"
        #ckpt_path = os.path.join(ROOT, 'pretrained', 'carnatic_ft_50', ckpt_name)

        # ----- Initialize Model -----
        
        tcn = MultiTracker(
            n_filters=PARAMS["N_FILTERS"],
            n_dilations=PARAMS["N_DILATIONS"],
            kernel_size=PARAMS["KERNEL_SIZE"],
            dropout_rate=PARAMS["DROPOUT"]
        )

        model = PLTCN.load_from_checkpoint(
            ckpt_path,
            model=tcn,
            params=PARAMS
        )
        model = model.to(device)

        # Initialize trainer
        trainer = L.Trainer(
            max_epochs=PARAMS["N_EPOCHS"],
            accelerator=accelerator,
        )

        # ----- Test -----

        # Run test and log results
        trainer.test(model, test_loader, verbose=True)

        # ----- Compute Averages -----
        
        tmp_results_path = os.path.join(os.getcwd(), 'temp', 'results.pkl')
        if os.path.exists(tmp_results_path):
            results_pkl = pd.read_pickle(tmp_results_path)
        else:
            print(f"Results file {tmp_results_path} not found.")
            
        # Create main DataFrame
        df = pd.DataFrame(results_pkl)
        # add taala column using the track_id
        df['taala'] = df['track_id'].apply(lambda x: carn_tracks[x].taala if x in carn_tracks else '')
        # rearrage columns to have 'track_id' and 'taala' first
        df = df[['track_id', 'taala'] + [col for col in df.columns if col not in ['track_id', 'taala']]]
        
        # Compute overall average (excluding non-numeric columns)
        avg_metrics = df.drop(columns=['track_id', 'taala']).mean()
        avg_metrics['track_id'] = 'average'
        avg_metrics['taala'] = ''

        # Compute per-Taala averages
        taala_averages = []
        for taala, group in df.groupby('taala'):
            taala_avg = group.drop(columns=['track_id', 'taala']).mean()
            taala_avg['track_id'] = ''
            taala_avg['taala'] = taala
            taala_averages.append(taala_avg)

        # Convert list of Series to DataFrame
        taala_avg_df = pd.DataFrame(taala_averages)

        # Concatenate everything: original data + taala averages + overall average
        df_with_avg = pd.concat([df, pd.DataFrame([avg_metrics]), taala_avg_df], ignore_index=True)
        
        # ----- Save Results -----
        
        output_dir = os.path.join(ROOT, 'output', 'results', 'tcn')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, ckpt_name.replace('.ckpt', '.csv'))
        df_with_avg.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        #delete pkl file
        if os.path.exists(tmp_results_path):
            os.remove(tmp_results_path)
            print(f"Deleted temporary file: {tmp_results_path}")