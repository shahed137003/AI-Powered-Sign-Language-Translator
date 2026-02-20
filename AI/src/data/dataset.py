import numpy as np
import torch
from torch.utils.data import Dataset

class ASLDataset(Dataset):
    def __init__(self, files, masks, labels):
        self.files = files
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.load(self.files[idx])).float().transpose(0, 1)
        m = torch.from_numpy(np.load(self.masks[idx])).float()
        y = torch.tensor(self.labels[idx])
        return x, m, y