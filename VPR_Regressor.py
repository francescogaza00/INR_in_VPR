import torch
import torchvision.transforms.functional as TF
from Network import *
from Pretrained_net import *
import json
from Utils import *
import matplotlib.pyplot as plt
import numpy as np

class VPR_Regressor(torch.nn.Module):
    def __init__(
        self,
        load_weights=False,
        hypersphere=False
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.hypersphere = hypersphere
        # Main encoder
        # self.first_encoder = MegaLoc()
        # self.first_dim = 8448
        # for name, param in self.first_encoder.named_parameters():
        #     if not name.startswith("aggregator.linear"):
        #         param.requires_grad = False
        # self.first_encoder.aggregator.agg.dust_bin.requires_grad = False

        self.first_encoder = CosPlace()
        self.first_dim = 2048
        for param in self.first_encoder.parameters():
            param.requires_grad = False  
                     
        # Input dimension for MLP
        self.input_dim = self.first_dim

        if self.hypersphere:
            self.mlp = MLP_hypersphere(input_dim=self.input_dim)
        else:
            self.mlp = MLP(input_dim=self.input_dim)

        if load_weights:
            weights_path = "mlp.pth"
            self.mlp.load_state_dict(torch.load(weights_path, weights_only=True))
        
        # if not self.hypersphere:
        #     self.alfa = nn.Parameter(torch.tensor(1.0))
        #     self.beta = nn.Parameter(torch.tensor(1.7))

    def forward(self, imgs):
        # feats, p_probs = self.first_encoder(imgs)   # for megaloc
        feats = self.first_encoder(imgs)  # for cosplace
        preds = self.mlp(feats)
        return preds

    def get_trainable_parameters(self):
        params = []
        params += list(self.mlp.parameters())
        # params += [self.first_encoder.aggregator.agg.dust_bin]
        return params

    def save_mlp_weights(self, path):
        torch.save(self.mlp.state_dict(), path)
    
    def load_mlp_weights(self, path):
        self.mlp.load_state_dict(torch.load(path, weights_only=True))
