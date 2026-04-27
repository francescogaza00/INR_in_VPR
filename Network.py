import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from Constants import M

_logger = logging.getLogger(__name__)


class _MLPTrunk(nn.Module):
    """Shared 3-layer hidden trunk used by all MLP variants."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act(self.fc3(x))


class MLPScaled(_MLPTrunk):
    """SCALED regression: 3-dim output in [0, 1] via sigmoid (lat/lon/bear each scaled to [0,1])."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.head = nn.Linear(input_dim, 3)
        _logger.info(f"MLPScaled: input_dim={input_dim}, output_dim=3 (sigmoid → [0,1])")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(super().forward(x)))


class MLPSphere3D(_MLPTrunk):
    """SPHERE3D regression: outputs [x, y, z, bear/360].

    A single joint head produces all 4 values so the final linear layer can
    learn the position-bearing correlation.  Constraints are then applied
    per-slice: L2-norm on the first 3 (→ S²), sigmoid on the last (→ [0,1]).
    """
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.head = nn.Linear(input_dim, 4)
        _logger.info(f"MLPSphere3D: input_dim={input_dim}, output_dim=4 (3 S²-normalised + 1 sigmoid)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.head(super().forward(x))
        xyz  = F.normalize(out[..., :3], p=2, dim=-1)  # unit vector on S²
        bear = torch.sigmoid(out[..., 3:])              # ∈ [0, 1]
        return torch.cat([xyz, bear], dim=-1)           # (B, 4)


class MLPHypersphere(_MLPTrunk):
    """HYPERSPHERE regression: outputs [w, x, y, z] L2-normalised on S³ (w²+x²+y²+z²=1)."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.head = nn.Linear(input_dim, 4)
        _logger.info(f"MLPHypersphere: input_dim={input_dim}, output_dim=4 (S³-normalised)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(super().forward(x)), p=2, dim=-1)


class MLPClassifier(_MLPTrunk):
    """Classification head for SOFT/HARD label modes: M²×12 raw logits (no softmax)."""
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.head = nn.Linear(input_dim, M * M * 12)
        _logger.info(f"MLPClassifier: input_dim={input_dim}, output_dim={M * M * 12}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(super().forward(x))
