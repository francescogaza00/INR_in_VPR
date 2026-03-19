import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
import random
import numpy as np
from Eval import *
from Utils import *
from VPR_Regressor import *
import sys
import time
import csv
import torch.nn as nn



def train_loop(
    train_dataloader: DataLoader,
    val_dataloader=None,
    test_dataloader=None,
    epochs=10,
    device='cuda',
    patience=10,
    model_save=False,
    load_weights=False,
    csvfile=None,
    hypersphere=False
    ):
    
    device = torch.device(device)

    model = VPR_Regressor(
        load_weights=load_weights
        )

    model.to(device)

    model.mlp.train()

    optimizer = Adam(model.get_trainable_parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,       
        factor=0.1,       
        patience=5,       
    )

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    best_val_loss = float('inf')
    bad_epochs = 0

    writer = csv.writer(csvfile,  delimiter=';')
    writer.writerow([
        "Epoch", "Train_Loss", "Val_Loss", "Time_min"
    ])


    for epoch in range(epochs):
        model.mlp.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        counter = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            imgs, targets = batch
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(imgs)

            #loss
            loss = loss_fn(preds, targets)

            loss.backward()
            
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time

        if val_dataloader is not None:
            avg_val_loss = evaluation_loop(model, val_dataloader, loss_fn, device, hypersphere=hypersphere)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Position Loss: {avg_val_loss:.6f} - Time: {epoch_time/60:.2f}minuti")
            print("")
            print("-" * 100)
            print("")
            writer.writerow([
                epoch+1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{epoch_time/60:.2f}"
            ])
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                bad_epochs = 0
                if model_save:
                    model.save_mlp_weights(f"mlp.pth")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping attivato. Training interrotto.")
                    break
        
        scheduler.step(avg_val_loss)
    
    print("LR:", current_lr)
    print("Training completato.")

    if val_dataloader is not None:
        visualize_predictions(model, val_dataloader, device, model_save=model_save, csvfile=csvfile)

    if test_dataloader is not None:
        visualize_predictions(model, test_dataloader, device, model_save=model_save, csvfile=csvfile)