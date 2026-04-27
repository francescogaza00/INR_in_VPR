import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
import random
import numpy as np
from Eval import *
from Utils import *
from VPR_Regressor import *
from Constants import LabelMode, RegressionEncoding
import sys
import time
import torch.nn as nn
import logging
import wandb

logger = logging.getLogger(__name__)



def train_loop(
    train_dataloader: DataLoader,
    val_dataloader=None,
    test_dataloader=None,
    epochs=10,
    device='cuda',
    patience=10,
    model_save=False,
    load_weights=False,
    backbone="cosplace",
    label_mode=LabelMode.REGRESSION,
    reg_encoding=RegressionEncoding.SCALED,
    ):
    """
    Training loop with validation, early stopping, and optional model saving.
    Metrics are logged to wandb every epoch; the best checkpoint is synced via wandb.save.

    Args:
        train_dataloader: DataLoader for training data.
        val_dataloader:   DataLoader for validation data. If None, skipped.
        test_dataloader:  DataLoader for test data. If None, skipped.
        epochs:           Maximum number of training epochs.
        device:           Target device ('cuda' or 'cpu').
        patience:         Epochs without val improvement before early stopping.
        model_save:       If True, saves best weights and syncs checkpoint to wandb.
        load_weights:     If True, restores pretrained MLP weights before training.
        backbone:         Backbone architecture name ('cosplace' or 'megaloc').
        reg_encoding:     How regression labels are encoded (SCALED / SPHERE3D / HYPERSPHERE).
    """
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(device)
    logger.info(f"Initializing model on {device} (backbone={backbone}, load_weights={load_weights}, label_mode={label_mode.name}, reg_encoding={reg_encoding.name if label_mode is LabelMode.REGRESSION else 'N/A'})")
    model  = VPR_Regressor(load_weights=load_weights, backbone=backbone, label_mode=label_mode, reg_encoding=reg_encoding)
    model.to(device)

    optimizer = Adam(model.get_trainable_parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    if label_mode is LabelMode.REGRESSION:
        _mse = nn.MSELoss(reduction='mean')
        if reg_encoding is RegressionEncoding.SCALED:
            loss_fn = lambda pred, tgt: _mse(pred, lat_lon_bear_scaler(tgt))
        elif reg_encoding is RegressionEncoding.SPHERE3D:
            loss_fn = lambda pred, tgt: _mse(pred, sphere3d_encoder(tgt))
        else:  # HYPERSPHERE
            loss_fn = lambda pred, tgt: _mse(pred, angles_to_hypersphere(tgt[..., 0], tgt[..., 1], tgt[..., 2]))
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

    best_val_loss = float('inf')
    bad_epochs    = 0

    logger.info(f"Starting training: max_epochs={epochs}, patience={patience}")

    for epoch in range(epochs):
        # Keep frozen backbone in eval mode; only the MLP head trains
        model.first_encoder.eval()
        model.mlp.train()
        epoch_start_time = time.time()
        running_loss = 0.0

        # --- Training ---
        for batch in train_dataloader:
            imgs, targets = batch
            imgs    = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = loss_fn(model(imgs), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        epoch_time     = time.time() - epoch_start_time
        current_lr     = optimizer.param_groups[0]['lr']

        # --- Validation ---
        if val_dataloader is not None:
            avg_val_loss = evaluation_loop(
                model, val_dataloader, loss_fn, device
            )
            scheduler.step(avg_val_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time/60:.2f}min"
            )

            wandb.log({
                "epoch":          epoch + 1,
                "train_loss":     avg_train_loss,
                "val_loss":       avg_val_loss,
                "lr":             current_lr,
                "epoch_time_min": epoch_time / 60,
            })

            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                bad_epochs    = 0
                if model_save:
                    model.save_mlp_weights("mlp.pth")
                    wandb.save("mlp.pth")  # sync best checkpoint to wandb
                    logger.info(f"New best val loss {best_val_loss:.6f} — checkpoint saved")
            else:
                bad_epochs += 1
                logger.info(f"No improvement ({bad_epochs}/{patience})")
                if bad_epochs >= patience:
                    logger.warning("Early stopping triggered.")
                    break

    logger.info(f"Training complete. Final LR: {current_lr:.2e}")

    # --- Final evaluation with prediction visualisation ---
    if val_dataloader is not None:
        logger.info("Running final evaluation on validation set")
        visualize_predictions(model, val_dataloader,  device,
                              load_best_weights=model_save, use_wandb=True,
                              label_mode=label_mode, reg_encoding=reg_encoding)
    if test_dataloader is not None:
        logger.info("Running final evaluation on test set")
        visualize_predictions(model, test_dataloader, device,
                              load_best_weights=model_save, use_wandb=True,
                              label_mode=label_mode, reg_encoding=reg_encoding)