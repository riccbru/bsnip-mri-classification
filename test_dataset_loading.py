# test_dataset_loading.py

from adni_dataset import ADNIDataset
import torch

# Load dataset from updated metadata
dataset = ADNIDataset('adni_preprocessed_metadata.csv')

print(f"✅ Total preprocessed samples: {len(dataset)}")

# Inspect a few samples
for i in range(3):
    img, label = dataset[i]
    print(f"Sample {i}: shape={img.shape}, label={label}, dtype={img.dtype}")
    assert isinstance(img, torch.Tensor) and img.shape == (1, 128, 128, 128)
