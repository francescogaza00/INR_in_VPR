# Main

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from Get_dataset import GeoDataset, LabelMode
from Constants import RegressionEncoding
from Training import *
import time
import logging
from Utils import *
from VPR_Regressor import *
import argparse
import wandb

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("outputs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Dataset root ──────────────────────────────────────────────────────────────
DATA_ROOT = "/home/gberton/data/vg_datasets/retrieval/sf_xl/processed"

# ── Worker seed function ──────────────────────────────────────────────────────
def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

g = torch.Generator()
g.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment",   action="store_true", help="Enable data augmentation")
    parser.add_argument("--no-test", action="store_false", dest="use_test", help="Disable test set evaluation")
    parser.add_argument("--backbone", type=str, default="cosplace", choices=["cosplace", "megaloc"])
    parser.add_argument(
        "--label-mode",
        choices=["soft", "hard", "regression"],
        default="regression",
        help="Label format: soft (IDW distribution), hard (cardinal class), regression (lat/lon/bear)",
    )
    parser.add_argument(
        "--reg-encoding",
        choices=["scaled", "sphere3d", "hypersphere"],
        default="scaled",
        help="Regression label encoding (only used with --label-mode regression): "
             "scaled (lat_lon_bear_scaler, 3-dim), "
             "sphere3d (latlon_to_unitvec + bear/360, 4-dim), "
             "hypersphere (angles_to_hypersphere, 4-dim S³ unit vector)",
    )
    opts = parser.parse_args()

    data_augmentation = opts.augment
    label_mode = {
        "soft":       LabelMode.SOFT,
        "hard":       LabelMode.HARD,
        "regression": LabelMode.REGRESSION,
    }[opts.label_mode]
    reg_encoding = {
        "scaled":      RegressionEncoding.SCALED,
        "sphere3d":    RegressionEncoding.SPHERE3D,
        "hypersphere": RegressionEncoding.HYPERSPHERE,
    }[opts.reg_encoding]

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data augmentation: {data_augmentation}")
    logger.info(f"Label mode: {label_mode.name}")
    if label_mode is LabelMode.REGRESSION:
        logger.info(f"Regression encoding: {reg_encoding.name}")

    # ── Dataset paths ─────────────────────────────────────────────────────────
    root_dir_train  = f"{DATA_ROOT}/train"
    root_dir_val_db = f"{DATA_ROOT}/val/database"
    root_dir_val_q  = f"{DATA_ROOT}/val/queries"
    root_dir_test   = f"{DATA_ROOT}/test"

    # ── Datasets and DataLoaders ──────────────────────────────────────────────
    train_dataset = GeoDataset(
        [root_dir_train],
        label_mode=label_mode,
        data_augmentation=data_augmentation,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=12,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_dataset = GeoDataset(
        [root_dir_val_db, root_dir_val_q],
        label_mode=label_mode,
    )
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=g,
    )

    if opts.use_test:
        test_dataset = GeoDataset(
            [root_dir_test],
            label_mode=label_mode,
        )
        logger.info(f"Test dataset size: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        test_dataloader = None
        logger.info("Test set disabled — skipping")

    # ── wandb init ────────────────────────────────────────────────────────────
    wandb.init(
        project="INR_in_VPR",
        config={
            "backbone":          opts.backbone,
            "label_mode":        opts.label_mode,
            "reg_encoding":      opts.reg_encoding,
            "data_augmentation": data_augmentation,
            "seed":              SEED,
            "epochs":            100,
            "batch_size_train":  4,
            "lr":                0.0001,
            "patience":          10,
            "train_size":        len(train_dataset),
            "val_size":          len(val_dataset),
            "test_size":         len(test_dataset) if test_dataloader is not None else None,
        },
    )

    # ── Training ──────────────────────────────────────────────────────────────
    start_time = time.time()
    train_loop(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        epochs=100,
        device=device,
        model_save=True,
        load_weights=False,
        backbone=opts.backbone,
        label_mode=label_mode,
        reg_encoding=reg_encoding,
    )
    elapsed = (time.time() - start_time) / 60
    logger.info(f"Training completed in {elapsed:.2f} minutes")
    wandb.log({"training_time_min": elapsed})
    wandb.finish()
