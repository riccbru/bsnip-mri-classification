import pandas as pd
import os
from datetime import datetime

# Load original metadata
df = pd.read_csv('adni_matched_nii_with_labels.csv')

npy_dir = './preprocessed_npy'

# Match .npy filenames based on same naming logic
def build_npy_path(row, i):
    rid = row['rid']
    scan_date = pd.to_datetime(row['scan_date']).strftime('%Y%m%d')
    filename = f"sub-{rid}_date-{scan_date}_i-{i}.npy"
    return os.path.join(npy_dir, filename)

# Add column for .npy path
df['npy_path'] = [build_npy_path(row, i) for i, row in df.iterrows()]

# Optional: keep only needed columns
df_out = df[['npy_path', 'rid', 'scan_date', 'diagnosis']]

df_out.to_csv('adni_preprocessed_npy_metadata.csv', index=False)
print(f"✅ Saved: adni_preprocessed_npy_metadata.csv with {len(df_out)} entries.")
