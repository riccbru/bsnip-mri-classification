# adni_dataset.py

import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np

class ADNIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.data = self.data[self.data['diagnosis'].isin([1.0, 2.0, 3.0])]
        self.data['label'] = self.data['diagnosis'].astype(int) - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.load(row['npy_path']).astype(np.float32)

        # Already normalized, but add channel dim
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img), torch.tensor(row['label'])

