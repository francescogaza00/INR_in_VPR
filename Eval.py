import torch
import torch.nn.functional as F
from Utils import *

def evaluation_loop(model, val_loader, loss_fn, device):
    val_running = 0.0

    model.eval()
    try:
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                val_running += loss_fn(model(imgs), targets).item()
    finally:
        model.train()

    return val_running / len(val_loader)