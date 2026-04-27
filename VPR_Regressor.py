import logging
import torch
from Network import *
from Pretrained_net import *
from Constants import LabelMode, RegressionEncoding

logger = logging.getLogger(__name__)

BACKBONE_DIMS = {
    "cosplace": 2048,
    "megaloc":  8448,
}

class VPR_Regressor(torch.nn.Module):
    def __init__(self, load_weights=False, backbone="cosplace",
                 label_mode=LabelMode.REGRESSION,
                 reg_encoding=RegressionEncoding.SCALED):
        super().__init__()

        assert backbone in BACKBONE_DIMS, f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONE_DIMS.keys())}"

        self.backbone      = backbone
        self.label_mode    = label_mode
        self.reg_encoding  = reg_encoding

        if backbone == "cosplace":
            self.first_encoder = CosPlace()
            for param in self.first_encoder.parameters():
                param.requires_grad = False
            logger.info("CosPlace backbone loaded — all parameters frozen")

        elif backbone == "megaloc":
            self.first_encoder = MegaLoc()
            for name, param in self.first_encoder.named_parameters():
                if not name.startswith("aggregator.linear"):
                    param.requires_grad = False
            self.first_encoder.aggregator.agg.dust_bin.requires_grad = False
            logger.info("MegaLoc backbone loaded — only aggregator.linear is trainable")

        dim = BACKBONE_DIMS[backbone]
        if label_mode is LabelMode.REGRESSION:
            if reg_encoding is RegressionEncoding.SCALED:
                self.mlp = MLPScaled(dim)
            elif reg_encoding is RegressionEncoding.SPHERE3D:
                self.mlp = MLPSphere3D(dim)
            else:  # HYPERSPHERE
                self.mlp = MLPHypersphere(dim)
        else:  # SOFT or HARD
            self.mlp = MLPClassifier(dim)
        logger.info(f"VPR_Regressor: label_mode={label_mode.name}, reg_encoding={reg_encoding.name if label_mode is LabelMode.REGRESSION else 'N/A'}, mlp={type(self.mlp).__name__}")

        if load_weights:
            self.load_mlp_weights("mlp.pth")

    def forward(self, imgs):
        if self.backbone == "megaloc":
            feats, _ = self.first_encoder(imgs)
        else:
            feats = self.first_encoder(imgs)
        return self.mlp(feats)

    def get_trainable_parameters(self):
        params = list(self.mlp.parameters())
        # if self.backbone == "megaloc":
        #     params += [self.first_encoder.aggregator.agg.dust_bin]
        return params

    def save_mlp_weights(self, path):
        torch.save(self.mlp.state_dict(), path)
        logger.info(f"MLP weights saved to '{path}'")

    def load_mlp_weights(self, path):
        self.mlp.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        logger.info(f"MLP weights loaded from '{path}'")
