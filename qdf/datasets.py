#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pickle
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        paths = sorted(Path(self.directory).iterdir(), key=os.path.getmtime)
        self.files = [str(p).strip().split('/')[-1] for p in paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.directory + self.files[idx], allow_pickle=True)


def mydataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = torch.utils.data.DataLoader(
                 dataset, batch_size, shuffle=shuffle, num_workers=num_workers,
                 collate_fn=lambda xs: list(zip(*xs)), pin_memory=True)
    return dataloader