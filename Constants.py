from enum import Enum, auto

LAT_MIN = 37.7575  # 37.7525
LAT_MAX = 37.7625 #37.7675
LON_MIN = -122.4325  # -122.4375
LON_MAX= -122.4275 # -122.4225
M = 10


class RegressionEncoding(Enum):
    """How (lat, lon, bearing) are encoded when label_mode=REGRESSION.

    - SCALED      – lat_lon_bear_scaler → 3-dim [0,1] vector.
    - SPHERE3D    – latlon_to_unitvec(lat,lon) + bear/360  → 4-dim vector.
    - HYPERSPHERE – angles_to_hypersphere(lat,lon,bear)    → 4-dim unit vector on S³.
    """
    SCALED      = auto()
    SPHERE3D    = auto()
    HYPERSPHERE = auto()


class LabelMode(Enum):
    """Controls what GeoDataset returns as ground truth.

    - SOFT       – float32 vector of length M*M*12, IDW-weighted over the 3×3 neighbourhood.
    - HARD       – scalar torch.long with the cardinal class index.
    - REGRESSION – float32 tensor [lat, lon, bearing] with raw values from the filename.
    """
    SOFT       = auto()
    HARD       = auto()
    REGRESSION = auto()