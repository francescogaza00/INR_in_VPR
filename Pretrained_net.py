import torch
from Megaloc import MegaLocModel
import torch.nn as nn



def MegaLoc() -> torch.nn.Module:
    model = MegaLocModel()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch", map_location="cpu")
    )
    model.eval()
    return model

def CosPlace():
    model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet101", fc_output_dim=2048)
    model.eval()
    return model

