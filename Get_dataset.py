import logging
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from Constants import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, M, LabelMode
from Utils import get_class

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def _build_soft_label(
    norm_lat: float,
    norm_lon: float,
    class_lat: int,
    class_lon: int,
    class_bear: int,
    n_classes: int,
) -> np.ndarray:
    """Float32 probability vector, IDW-weighted over the 3×3 neighbourhood."""
    label = np.zeros(n_classes, dtype=np.float32)
    pos = np.array([norm_lat * M, norm_lon * M], dtype=np.float32)

    neighbors = [
        (class_lat + di, class_lon + dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if 0 <= class_lat + di < M and 0 <= class_lon + dj < M
    ]

    centers = np.array([[ni + 0.5, nj + 0.5] for ni, nj in neighbors], dtype=np.float32)
    dists = np.linalg.norm(pos - centers, axis=1)

    if np.any(dists == 0):
        best = int(np.argmin(dists))
        ni, nj = neighbors[best]
        label[get_class(ni, nj, class_bear)] = 1.0
    else:
        weights = 1.0 / dists
        weights /= weights.sum()
        for w, (ni, nj) in zip(weights, neighbors):
            label[get_class(ni, nj, class_bear)] = w

    return label


def _build_hard_label(class_lat: int, class_lon: int, class_bear: int) -> int:
    """Cardinal class index for the exact cell."""
    return get_class(class_lat, class_lon, class_bear)


def _build_regression_label(lat: float, lon: float, bearing: int) -> np.ndarray:
    """Raw float32 array ``[lat, lon, bearing]``."""
    return np.array([lat, lon, float(bearing)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GeoDataset(Dataset):
    """Geo-localisation dataset.

    Scans *root_dirs* recursively for ``.jpg`` images whose filenames encode
    GPS coordinates and camera bearing in the ``@``-separated format::

        ..@..@..@..@..@<lat>@<lon>@..@..@<bearing>@...jpg

    The format of the returned ground truth is controlled by *label_mode*:

    +-----------------------+--------+------------------------------------------+
    | ``LabelMode``         | dtype  | shape / content                          |
    +=======================+========+==========================================+
    | ``SOFT`` (default)    | float32| ``(M*M*12,)`` IDW distribution           |
    +-----------------------+--------+------------------------------------------+
    | ``HARD``              | long   | scalar cardinal class index              |
    +-----------------------+--------+------------------------------------------+
    | ``REGRESSION``        | float32| ``(3,)`` → ``[lat, lon, bearing]``       |
    +-----------------------+--------+------------------------------------------+

    Args:
        root_dirs:         List of root directories to scan.
        label_mode:        One of ``LabelMode.SOFT``, ``LabelMode.HARD``,
                           ``LabelMode.REGRESSION``.  Default: ``LabelMode.SOFT``.
        data_augmentation: If ``True``, applies ``train_augmented_transforms``
                           (colour jitter + Gaussian blur) instead of the
                           standard ``train_transforms``.  Default: ``False``.
    """

    _LAT_IDX  = 5
    _LON_IDX  = 6
    _BEAR_IDX = 9

    def __init__(
        self,
        root_dirs: list[str],
        label_mode: LabelMode = LabelMode.REGRESSION,
        data_augmentation: bool = False,
    ) -> None:
        self.root_dirs = root_dirs
        self.label_mode = label_mode
        self.data_augmentation = data_augmentation
        self.transform = train_augmented_transforms if data_augmentation else train_transforms

        self.image_paths: list[str] = []
        self.ground_truths: list = []

        self._scan_directories()

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in the provided directories: {root_dirs}"
            )

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_directories(self) -> None:
        """Populate ``image_paths`` and ``ground_truths`` by walking *root_dirs*."""
        n_classes = M * M * 12
        missing = skipped = errors = 0

        for root_dir in self.root_dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                folder_name = os.path.basename(dirpath)
                if self._is_out_of_lat_range(folder_name):
                    continue

                for filename in filenames:
                    if not filename.lower().endswith(".jpg"):
                        continue

                    img_path = os.path.join(dirpath, filename)
                    try:
                        lat, lon, bearing = self._parse_filename(filename)
                    except ValueError:
                        missing += 1
                        continue
                    except Exception:
                        errors += 1
                        continue

                    # Bounding-box filter — always applied regardless of label mode
                    norm_lat = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
                    norm_lon = (lon - LON_MIN) / (LON_MAX - LON_MIN)

                    if not (0.0 <= norm_lat < 1.0 and 0.0 <= norm_lon < 1.0):
                        skipped += 1
                        continue

                    # REGRESSION mode: no class computation needed
                    if self.label_mode is LabelMode.REGRESSION:
                        self.image_paths.append(img_path)
                        self.ground_truths.append(
                            _build_regression_label(lat, lon, bearing)
                        )
                        continue

                    if bearing == 360:
                        bearing = 0

                    class_lat  = int(norm_lat * M)
                    class_lon  = int(norm_lon * M)
                    class_bear = int(bearing / 30)

                    if not (class_lat < M and class_lon < M and class_bear < 12):
                        logger.warning(
                            f"Skipping '{filename}': class out of range "
                            f"(lat={class_lat}, lon={class_lon}, bear={class_bear})"
                        )
                        skipped += 1
                        continue

                    if self.label_mode is LabelMode.SOFT:
                        label = _build_soft_label(
                            norm_lat, norm_lon, class_lat, class_lon, class_bear, n_classes
                        )
                    else:  # HARD
                        label = _build_hard_label(class_lat, class_lon, class_bear)

                    self.image_paths.append(img_path)
                    self.ground_truths.append(label)

        logger.info(
            f"Dataset scan complete: loaded={len(self.image_paths)}, "
            f"missing={missing}, skipped={skipped}, errors={errors}, "
            f"mode={self.label_mode.name}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_out_of_lat_range(folder_name: str) -> bool:
        try:
            lat = float(folder_name)
            return not (LAT_MIN <= lat <= LAT_MAX)
        except ValueError:
            return False

    @staticmethod
    def _parse_filename(filename: str) -> tuple[float, float, int]:
        """Extract ``(lat, lon, bearing)`` from a ``@``-separated filename."""
        stem = filename.strip().lower().removesuffix(".jpg")
        parts = stem.split("@")
        try:
            lat     = float(parts[GeoDataset._LAT_IDX])
            lon     = float(parts[GeoDataset._LON_IDX])
            bearing = int(parts[GeoDataset._BEAR_IDX])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Cannot parse '{filename}': {exc}") from exc
        return lat, lon, bearing

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        raw      = self.ground_truths[idx]

        if self.label_mode is LabelMode.HARD:
            label = torch.tensor(raw, dtype=torch.long)
        else:  # SOFT or REGRESSION → float32
            label = torch.tensor(raw, dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_RESIZE        = (252, 252)

train_transforms = transforms.Compose([
    transforms.Resize(_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

train_augmented_transforms = transforms.Compose([
    transforms.Resize(_RESIZE),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize(_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])
