import torch
import torch.nn as nn
import random
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
import numpy as np
from Constants import *
import csv


def visualize_predictions(model, dataloader, device, num_samples=20, model_save=False, csvfile=None, hypersphere=False):
    if model_save:
        weights_path = "mlp.pth"
        model.load_mlp_weights(weights_path)
    model.eval()
    shown = 0
    correct = 0
    csv_rows = []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            preds = torch.argmax(preds, dim=1) 
            preds = preds.int()
            preds = preds.cpu().numpy()
            targets = torch.argmax(targets, dim=1)
            targets = targets.cpu().numpy()
            for i in range(len(imgs)):
                if preds[i] == targets[i]:
                    correct += 1


    total = len(dataloader.dataset)
    if csvfile is not None:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(csv_rows)
        writer.writerow(["Accuracy", correct, total, f"{correct / total * 100:.2f}%"])

    print(f"Accuracy: {correct}/{total} = {correct / total * 100:.2f}%")


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
    lat = torch.deg2rad(lat_deg)
    lon = torch.deg2rad(lon_deg)
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)

def unitvec_to_latlon(vec):

    v = torch.as_tensor(vec)

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


def lat_lon_bear_scaler(vec: torch.Tensor):

    if vec.shape[-1] != 3:
        raise ValueError("Last dimension must be of size 3 (latitude, longitude, bearing).")
    
    lat = vec[..., 0]
    lon = vec[..., 1]
    bear = vec[..., 2]

    lat_scaled = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_scaled = (lon - LON_MIN) / (LON_MAX - LON_MIN)
    bear_scaled = bear / 360.0

    return torch.stack([lat_scaled, lon_scaled, bear_scaled], dim=-1)

def lat_lon_bear_unscaler(vec: torch.Tensor):

    if vec.shape[-1] != 3:
        raise ValueError("Last dimension must be of size 3 (latitude, longitude, bearing).")
    
    lat_scaled = vec[..., 0]
    lon_scaled = vec[..., 1]
    bear_scaled = vec[..., 2]

    lat = lat_scaled * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_scaled * (LON_MAX - LON_MIN) + LON_MIN
    bear = bear_scaled * 360.0

    return torch.stack([lat, lon, bear], dim=-1)

def angles_to_hypersphere(lat, lon, bearing):
    lat = torch.deg2rad(lat)
    lon = torch.deg2rad(lon)
    bearing = torch.deg2rad(bearing)
    w = torch.sin(lat)
    x = torch.cos(lat) * torch.cos(lon) * torch.cos(bearing)
    y = torch.cos(lat) * torch.cos(lon) * torch.sin(bearing)
    z = torch.cos(lat) * torch.sin(lon)
    return torch.stack([w, x, y, z], dim=-1)


def hypersphere_to_angles(vec4):
    v = torch.as_tensor(vec4, dtype=torch.float64)
    if v.shape[-1] != 4:
        raise ValueError("Last dim must be 4 (w,x,y,z).")

    w = v[..., 0]
    x = v[..., 1]
    y = v[..., 2]
    z = v[..., 3]

    w_clipped = torch.clamp(w, -1.0, 1.0)

    lat = torch.asin(w_clipped)   # NB lat in [-pi/2, pi/2]

    bearing = torch.atan2(y, x)

    r = torch.hypot(x, y)

    lon = torch.atan2(z, r)

    lat = torch.rad2deg(lat)
    lon = torch.rad2deg(lon)
    # lon = (180.0 - lon) % 360.0 - 360
    bearing = torch.rad2deg(bearing)
    bearing = (bearing + 360.0) % 360.0
    # bearing = (bearing + 180.0) % 360.0

    
    return torch.stack([lat, lon, bearing], dim=-1)

def lat_lon_scaler(vec: torch.Tensor):

    if vec.shape[-1] != 2:
        raise ValueError("Last dimension must be of size 2 (latitude, longitude).")
    
    lat = vec[..., 0]
    lon = vec[..., 1]

    lat_scaled = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
    lon_scaled = (lon - LON_MIN) / (LON_MAX - LON_MIN)

    return torch.stack([lat_scaled, lon_scaled], dim=-1)

def lat_lon_unscaler(vec: torch.Tensor):

    if vec.shape[-1] != 2:
        raise ValueError("Last dimension must be of size 2 (latitude, longitude).")
    
    lat_scaled = vec[..., 0]
    lon_scaled = vec[..., 1]

    lat = lat_scaled * (LAT_MAX - LAT_MIN) + LAT_MIN
    lon = lon_scaled * (LON_MAX - LON_MIN) + LON_MIN

    return torch.stack([lat, lon], dim=-1)