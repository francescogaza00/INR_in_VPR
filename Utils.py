import torch
import numpy as np
from Constants import *  # includes LabelMode, RegressionEncoding
import wandb

# ImageNet normalisation constants — used to denormalise images before display
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

_CORRECT_THRESHOLD_M = 25.0  # regression "correct" = prediction within this distance


def visualize_predictions(model, dataloader, device, num_samples=20,
                          load_best_weights=False, use_wandb=False,
                          label_mode=None, reg_encoding=None):
    if label_mode is None:
        label_mode = LabelMode.REGRESSION
    if reg_encoding is None:
        reg_encoding = RegressionEncoding.SCALED

    if load_best_weights:
        model.load_mlp_weights("mlp.pth")

    model.eval()
    correct = 0
    total_err_m = 0.0
    shown   = 0

    if use_wandb:
        table = wandb.Table(columns=[
            "image", "pred_class", "target_class",
            "pred_lat", "pred_lon", "target_lat", "target_lon",
            "error_m", "correct",
        ])

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            raw_preds = model(imgs)

            if label_mode is LabelMode.REGRESSION:
                # Decode predictions from encoded space back to (lat, lon, bearing)
                if reg_encoding is RegressionEncoding.SCALED:
                    decoded = lat_lon_bear_unscaler(raw_preds.cpu())
                elif reg_encoding is RegressionEncoding.SPHERE3D:
                    decoded = sphere3d_decoder(raw_preds.cpu())
                else:  # HYPERSPHERE
                    decoded = hypersphere_to_angles(raw_preds.cpu()).float()
                p_lats = decoded[:, 0].numpy()
                p_lons = decoded[:, 1].numpy()
                t_lats = targets[:, 0].cpu().numpy()
                t_lons = targets[:, 1].cpu().numpy()
                errs   = haversine_distance(p_lats, p_lons, t_lats, t_lons)

                for i in range(len(imgs)):
                    err_i       = float(errs[i])
                    is_correct  = err_i <= _CORRECT_THRESHOLD_M
                    correct    += int(is_correct)
                    total_err_m += err_i

                    if use_wandb and shown < num_samples:
                        img_disp = (imgs[i].cpu() * _IMAGENET_STD[:, None, None]
                                    + _IMAGENET_MEAN[:, None, None]).clamp(0, 1)
                        img_disp = img_disp.permute(1, 2, 0).numpy()
                        table.add_data(
                            wandb.Image(img_disp),
                            "—", "—",
                            round(float(p_lats[i]), 5), round(float(p_lons[i]), 5),
                            round(float(t_lats[i]), 5), round(float(t_lons[i]), 5),
                            round(err_i, 1), bool(is_correct),
                        )
                        shown += 1

            else:  # SOFT or HARD
                preds   = torch.argmax(raw_preds, dim=1).int().cpu().numpy()
                if label_mode is LabelMode.SOFT:
                    tgt_cls = torch.argmax(targets, dim=1).cpu().numpy()
                else:  # HARD — target is already a scalar class index
                    tgt_cls = targets.cpu().numpy()

                for i in range(len(imgs)):
                    is_correct = int(preds[i] == tgt_cls[i])
                    correct += is_correct

                    if use_wandb and shown < num_samples:
                        p_lat, p_lon = class_to_latlon(int(preds[i]))
                        t_lat, t_lon = class_to_latlon(int(tgt_cls[i]))
                        err = float(haversine_distance(p_lat, p_lon, t_lat, t_lon))

                        img_disp = (imgs[i].cpu() * _IMAGENET_STD[:, None, None]
                                    + _IMAGENET_MEAN[:, None, None]).clamp(0, 1)
                        img_disp = img_disp.permute(1, 2, 0).numpy()
                        table.add_data(
                            wandb.Image(img_disp),
                            int(preds[i]),   int(tgt_cls[i]),
                            round(p_lat, 5), round(p_lon, 5),
                            round(t_lat, 5), round(t_lon, 5),
                            round(err, 1),   bool(is_correct),
                        )
                        shown += 1

    total = len(dataloader.dataset)
    if label_mode is LabelMode.REGRESSION:
        mean_err = total_err_m / total
        recall   = correct / total
        print(
            f"Recall@{_CORRECT_THRESHOLD_M:.0f}m: {correct}/{total} = {recall * 100:.2f}% | "
            f"Mean error: {mean_err:.1f} m"
        )
        if use_wandb:
            wandb.log({
                f"recall_at_{_CORRECT_THRESHOLD_M:.0f}m": recall,
                "mean_error_m": mean_err,
                "predictions":  table,
            })
    else:
        accuracy = correct / total
        print(f"Accuracy: {correct}/{total} = {accuracy * 100:.2f}%")
        if use_wandb:
            wandb.log({"accuracy": accuracy, "predictions": table})


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcola la distanza fra due serie di punti geografici (lat1, lon1) e (lat2, lon2),
    tutti in gradi, restituendo un array di distanze in metri.
    
    lat1, lon1, lat2, lon2 possono essere scalari o array NumPy della stessa shape.
    """
    # Raggio medio della Terra in metri
    R = 6371000.0

    # Conversione in radianti
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)

    # Formula di Haversine
    a = np.sin(Δφ / 2.0)**2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c

def latlon_to_unitvec(lat_deg, lon_deg):
    """Convert (lat, lon) in degrees to a unit vector on the 3-D sphere."""
    if lat_deg.shape != lon_deg.shape:
        raise ValueError("lat_deg and lon_deg must have the same shape.")
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def unitvec_to_latlon(vec):
    """Inverse of latlon_to_unitvec: recover (lat, lon) in degrees from a unit vector."""
    v = torch.as_tensor(vec, dtype=torch.float64)

    if v.shape[-1] != 3:
        raise ValueError("Last dim must be 3 (x,y,z).")

    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    lat_rad = torch.asin(z)          # latitude in radians
    lon_rad = torch.atan2(y, x)      # longitude in radians

    lat = torch.rad2deg(lat_rad)
    lon = torch.rad2deg(lon_rad)

    return torch.stack([lat, lon], dim=-1)


def lat_lon_bear_scaler(vec: torch.Tensor) -> torch.Tensor:
    """Normalise (lat, lon, bearing) to [0, 1] using the dataset's geographic bounding box."""
    if vec.shape[-1] != 3:
        raise ValueError("Last dimension must be of size 3 (latitude, longitude, bearing).")

    lat = vec[..., 0]
    lon = vec[..., 1]
    bear = vec[..., 2]

    lat_scaled = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_scaled = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    bear_scaled = bear / 360.0  # bearing is already in [0, 360)

    return torch.stack([lat_scaled, lon_scaled, bear_scaled], dim=-1)


def lat_lon_bear_unscaler(vec: torch.Tensor) -> torch.Tensor:
    """Inverse of lat_lon_bear_scaler: map [0, 1] values back to original units."""
    if vec.shape[-1] != 3:
        raise ValueError("Last dimension must be of size 3 (latitude, longitude, bearing).")

    lat_scaled = vec[..., 0]
    lon_scaled = vec[..., 1]
    bear_scaled = vec[..., 2]

    lat = lat_scaled * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_scaled * (LON_MAX - LON_MIN) + LON_MIN
    bear = bear_scaled * 360.0

    return torch.stack([lat, lon, bear], dim=-1)


def angles_to_hypersphere(lat, lon, bearing) -> torch.Tensor:
    """
    Map (lat, lon, bearing) in degrees onto the unit 3-sphere S^3.
    Encoding: w=sin(lat), x=cos(lat)*cos(lon)*cos(bearing),
              y=cos(lat)*cos(lon)*sin(bearing), z=cos(lat)*sin(lon).
    Valid for lat in (-90, 90); lon recovery relies on cos(lat) > 0.
    """
    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)
    bearing = torch.deg2rad(bearing)
    w = torch.sin(lat)
    x = torch.cos(lat) * torch.cos(lon) * torch.cos(bearing)
    y = torch.cos(lat) * torch.cos(lon) * torch.sin(bearing)
    z = torch.cos(lat) * torch.sin(lon)
    return torch.stack([w, x, y, z], dim=-1)


def hypersphere_to_angles(vec4) -> torch.Tensor:
    """
    Inverse of angles_to_hypersphere: recover (lat, lon, bearing) in degrees.
    Note: lon recovery uses |cos(lat)| as denominator, which is correct when
    cos(lat) > 0, i.e. lat in (-90°, 90°) — always satisfied for San Francisco.
    """
    v = torch.as_tensor(vec4, dtype=torch.float64)
    if v.shape[-1] != 4:
        raise ValueError("Last dim must be 4 (w,x,y,z).")

    w = v[..., 0]
    x = v[..., 1]
    y = v[..., 2]
    z = v[..., 3]

    # clamp to [-1, 1] to handle minor floating-point overflows before asin
    lat = torch.asin(torch.clamp(w, -1.0, 1.0))

    bearing = torch.atan2(y, x)

    # r = cos(lat)*|cos(lon)|; valid inversion because cos(lat) > 0 here
    r = torch.hypot(x, y)
    lon = torch.atan2(z, r)

    lat = torch.rad2deg(lat)
    lon = torch.rad2deg(lon)
    bearing = torch.rad2deg(bearing)
    bearing = (bearing + 360.0) % 360.0  # normalise to [0, 360)

    return torch.stack([lat, lon, bearing], dim=-1)


def lat_lon_scaler(vec: torch.Tensor) -> torch.Tensor:
    """Normalise (lat, lon) to [0, 1] using the dataset's geographic bounding box."""
    if vec.shape[-1] != 2:
        raise ValueError("Last dimension must be of size 2 (latitude, longitude).")

    lat = vec[..., 0]
    lon = vec[..., 1]

    lat_scaled = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_scaled = (lon - LON_MIN) / (LON_MAX - LON_MIN)

    return torch.stack([lat_scaled, lon_scaled], dim=-1)


def lat_lon_unscaler(vec: torch.Tensor) -> torch.Tensor:
    """Inverse of lat_lon_scaler: map [0, 1] values back to (lat, lon) in degrees."""
    if vec.shape[-1] != 2:
        raise ValueError("Last dimension must be of size 2 (latitude, longitude).")

    lat_scaled = vec[..., 0]
    lon_scaled = vec[..., 1]

    lat = lat_scaled * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_scaled * (LON_MAX - LON_MIN) + LON_MIN

    return torch.stack([lat, lon], dim=-1)


def sphere3d_encoder(tgt: torch.Tensor) -> torch.Tensor:
    """Encode (lat, lon, bearing) → (x, y, z, bear/360), shape (..., 4)."""
    uvec        = latlon_to_unitvec(tgt[..., 0], tgt[..., 1])        # (..., 3)
    bear_scaled = (tgt[..., 2] / 360.0).unsqueeze(-1)                # (..., 1)
    return torch.cat([uvec, bear_scaled], dim=-1)                     # (..., 4)


def sphere3d_decoder(pred: torch.Tensor) -> torch.Tensor:
    """Decode (x, y, z, bear/360) → (lat, lon, bearing), shape (..., 3)."""
    latlon = unitvec_to_latlon(pred[..., :3]).float()                 # (..., 2)
    bear   = pred[..., 3:] * 360.0                                   # (..., 1)
    return torch.cat([latlon, bear], dim=-1)                          # (..., 3)


def get_class(class_lat: int, class_lon: int, class_bear: int) -> int:
    """
    Flatten (lat_cell, lon_cell, bearing_bin) into a single linear class index.
    Grid: M x M spatial cells, 12 bearing bins (30° each) → M*M*12 total classes.
    """
    return class_lat * (M * 12) + class_lon * 12 + class_bear


def class_to_latlon(class_idx: int) -> tuple[float, float]:
    """
    Inverse of get_class: recover the centre (lat, lon) of the spatial cell
    corresponding to a linear class index.
    """
    class_lat = class_idx // (M * 12)
    class_lon = (class_idx % (M * 12)) // 12
    lat = LAT_MIN + (class_lat + 0.5) / M * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (class_lon + 0.5) / M * (LON_MAX - LON_MIN)
    return lat, lon