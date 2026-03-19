import logging
import math
import re
import torch
import torch.nn as nn
from Constants import *
import torch.nn.functional as F


_logger = logging.getLogger(__name__)

class MLP_hypersphere(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)

        self.fc2 = nn.Linear(input_dim, 4)

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)

        self.fc2 = nn.Linear(input_dim, input_dim)

        self.fc3 = nn.Linear(input_dim, input_dim)

        self.fc4 = nn.Linear(input_dim, M*M*12)

        self.act = nn.SiLU()

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc4(x)
        return x


# class MLP_angle(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()

#         self.fc1 = nn.Linear(input_dim + 3, input_dim + 3)

#         self.fc2 = nn.Linear(input_dim, 2)

#         self.fc3 = nn.Linear(2 , 2)

#         self.act = nn.SiLU()

#         self.softplus = nn.Softplus()


#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.fc2(x)
#         x = F.normalize(x, p=2, dim=1)
#         return x

