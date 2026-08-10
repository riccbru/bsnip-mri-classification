import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom

# Load CSV with file paths
csv_path = 'adni_matched_nii_with_labels.csv'
df = pd.read_csv(csv_path)

# Desired output shape
output_shape = (128, 128, 128)
output_dir = './preprocessed_npy'
os.makedirs(output_dir, exist_ok=True)

def preprocess_volume(path, output_shape):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # Normalize: z-score
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    else:
        data = data - mean

    # Resize
    zoom_factors = [o / i for o, i in zip(output_shape, data.shape)]
    data_resized = zoom(data, zoom_factors, order=1)  # linear interpolation

    return data_resized

# Process and save
for i, row in tqdm(df.iterrows(), total=len(df)):
    nii_path = row['filepath']
    try:
        vol = preprocess_volume(nii_path, output_shape)
        rid = row['rid']
        scan_date = pd.to_datetime(row['scan_date']).strftime('%Y%m%d')
        out_name = f"sub-{rid}_date-{scan_date}_i-{i}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, vol)
    except Exception as e:
        print(f"❌ Failed: {nii_path} → {e}")

print("✅ All done. Preprocessed volumes saved to:", output_dir)
