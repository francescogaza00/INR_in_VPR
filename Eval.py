import torch
import torch.nn.functional as F
from Utils import *

def evaluation_loop(model, val_loader, loss_fn, device, hypersphere=False):
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)

            #loss
            loss = loss_fn(preds, targets)
            
            batch_loss = loss.item()
            val_running += batch_loss
    return val_running / len(val_loader)