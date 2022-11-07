#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class QDFDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str):
        self.directory = directory
        paths = sorted(Path(self.directory).iterdir(), key=os.path.getmtime)
        self.files = [str(p).strip().split("/")[-1] for p in paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.directory + self.files[idx], allow_pickle=True)
