# Main

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from Get_dataset import *
from Training import *
import time
import sys
from Utils import *
from VPR_Regressor import *
import csv


# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    # Redirect print on outputs.txt
    sys.stdout = open('outputs.txt', 'a')

    csv_path = "results_cosplace_resnet__output_.csv"
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set your dataset root directory here
    root_dir_train = "/home/gberton/data/vg_datasets/retrieval/sf_xl/processed/train"
    root_dir_val_db = "/home/gberton/data/vg_datasets/retrieval/sf_xl/processed/val/database"
    root_dir_val_q = "/home/gberton/data/vg_datasets/retrieval/sf_xl/processed/val/queries"
    root_dir_test = "/home/gberton/data/vg_datasets/retrieval/sf_xl/processed/test"

    # Create dataset and dataloader with optimizations
    data_augmentation = True
    train_dataset = get_dataset(
        [root_dir_train],
        transform=train_augmented_transforms if data_augmentation else train_transforms,
        data_augmentation=data_augmentation
        )
    print(f"Training dataset size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=12,
        pin_memory=True if device.type == "cuda" else False
    )

    val_dataset = get_dataset([root_dir_val_db, root_dir_val_q], transform=eval_transforms)
    print(f"Validation dataset size: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    test_dataset = get_dataset([root_dir_test], transform=eval_transforms)
    print(f"Test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # Prepara il file CSV
    with open(csv_path, "a", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["Training dataset size", len(train_dataset)])
        writer.writerow(["Validation dataset size", len(val_dataset)])
        # Training loop
        start_time = time.time()
        train_loop(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=None,
            epochs=100,
            device=device,
            model_save=True,
            load_weights=False,
            csvfile=csvfile
        )
        end_time = time.time()
        writer.writerow(["Info", "Tempo di training (minuti)", f"{(end_time - start_time)/60:.2f}"])