import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re
import random
import numpy as np
from Constants import *
from Utils import *

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class get_dataset(Dataset):
    
    def __init__(self, root_dirs, transform, data_augmentation=False):
        """
        Inizializza il dataset con la root directory e le trasformazioni RGB.
        Inizializza il dataset con la root directory e le trasformazioni RGB.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        self.ground_truths = []
        self.image_encoding = []
        self.data_augmentation = data_augmentation
        self._scan_directories()


        if not self.image_paths:
            raise RuntimeError(f"Nessuna immagine trovata nella directory: {root_dirs}")

    def _scan_directories(self):
        missing_values = 0
        error_values = 0
        """
        Scansiona ricorsivamente la root_dir per trovare tutti i percorsi delle immagini
        e popolare le liste self.image_paths e self.ground_truths.
        """
        for root_dir in self.root_dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                folder_name = os.path.basename(dirpath)
                # Prova a interpretare il nome della cartella come latitudine
                try:
                    folder_lat = float(folder_name)
                    # Salta la cartella se la latitudine è fuori dal range
                    if folder_lat < LAT_MIN or folder_lat > LAT_MAX:
                        continue
                except ValueError:
                    # Se il nome della cartella non è una latitudine valida passa alla prossima
                    pass
                for filename in filenames:
                    # Controlla se il file è un'immagine
                    if filename.lower().endswith('.jpg'):
                        img_path = os.path.join(dirpath, filename)
                        try:
                            # Estrai la ground truth dal nome del file
                            ground_truth = self.ground_truth_extraction(filename)
                            lat, lon, bearing = ground_truth
                            norm_lat = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
                            norm_lon = (lon - LON_MIN) / (LON_MAX - LON_MIN)
                            # Filtro
                            if 0 <= norm_lat < 1 and 0 <= norm_lon < 1: # and len(self.image_paths) < 2000:
                                # precision = 1 / M
                                # ground_truth = [lat, lon, bearing]
                                if bearing == 360:
                                    bearing = 0
                                class_lat = int(norm_lat * M)
                                class_lon = int(norm_lon * M)
                                class_bear = int(bearing / 30)
                                true_class = get_class(class_lat, class_lon, class_bear)
                                if class_lat < M and class_lon < M and class_bear < 12:
                                    # Build distribution over the current cell and its neighbors
                                    ground_truth = np.zeros(M * M * 12, dtype=np.float32)
                                    # position in normalized coordinates
                                    pos = np.array([norm_lat, norm_lon], dtype=np.float32)
                                    neighbors = []
                                    for di in (-1, 0, 1):
                                        for dj in (-1, 0, 1):
                                            ni = class_lat + di
                                            nj = class_lon + dj
                                            if 0 <= ni < M and 0 <= nj < M:
                                                neighbors.append((ni, nj))

                                    # compute distances to neighbor cell centers and convert to weights
                                    dists = []
                                    centers = []
                                    for (ni, nj) in neighbors:
                                        center = np.array([ni + 0.5, nj + 0.5], dtype=np.float32)
                                        centers.append((ni, nj, center))
                                        dists.append(np.linalg.norm(pos - center))

                                    dists = np.array(dists, dtype=np.float32)
                                    # if any distance is zero, give that cell full weight
                                    if np.any(dists == 0):
                                        zero_idx = int(np.argmin(dists))
                                        ni, nj, _ = centers[zero_idx]
                                        ground_truth[get_class(ni, nj, class_bear)] = 1.0
                                    else:
                                        # inverse distance weighting
                                        weights = 1.0 / (dists + 1e-8)
                                        weights = weights / np.sum(weights)
                                        for w, (ni, nj, _) in zip(weights, centers):
                                            ground_truth[get_class(ni, nj, class_bear)] = w

                                    self.image_paths.append(img_path)
                                    # ground_truth = [class_lat, class_lon, class_bear]
                                    # ground_truth = true_class
                                    self.ground_truths.append(ground_truth)
                                else:
                                    print(f"Skipping file '{filename}' due to class out of range: lat {class_lat}, lon {class_lon}, bear {class_bear}")
                        except ValueError as e:
                            missing_values += 1
                            # print(f"Skipping file '{filename}' due to parsing error: {e}")
                        except Exception as e:
                            # print(f"An unexpected error occurred with file '{filename}': {e}")
                            error_values += 1
        print('missing values:', missing_values)
        print('error values:', error_values)

    def ground_truth_extraction(self, filename):
        try:
            # Rimuove estensione se presente
            filename = filename.strip().lower()
            if filename.endswith('.jpg'):
                filename = filename[:-4]
            
            parts = filename.split('@')
            lat = float(parts[5])    
            lon = float(parts[6])
            bearing = int(parts[9])
            return (lat, lon, bearing)

        except (IndexError, ValueError) as e:
            raise ValueError(f"Impossibile estrarre i valori da '{filename}': {e}")

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        orig = self.ground_truths[idx]
        # ground_truth = [float(x) for x in orig]

        ground_truth = orig

        # if self.data_augmentation:
        #     for i in range(len(ground_truth)):
        #         ground_truth[i] += float(np.random.normal(0, 0.002))

        # # clamp
        # ground_truth[0] = min(max(ground_truth[0], LAT_MIN), LAT_MAX)
        # ground_truth[1] = min(max(ground_truth[1], LON_MIN), LON_MAX)
        # ground_truth[2] = min(max(ground_truth[2], 0.0), 360.0)

        # probabilities/weights -> float tensor
        ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)

        # Carica l'immagine usando PIL e convertila sempre in RGB inizialmente
        original_image = Image.open(img_path).convert('RGB')

        # Applica le trasformazioni
        image = self.transform(original_image)

        # Restituisce entrambe le versioni dell'immagine e la ground truth
        return image, ground_truth_tensor

train_transforms = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

train_augmented_transforms = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.RandomChoice([
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


eval_transforms = transforms.Compose([
    transforms.Resize((252, 252)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

def get_class(class_lat, class_lon, class_bear):
    return (class_lat) * (M*12) + (class_lon) * 12 + class_bear
