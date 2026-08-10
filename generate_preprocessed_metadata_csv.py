# generate_preprocessed_metadata_csv.py

import pandas as pd
import os

# Load original metadata
df = pd.read_csv('adni_matched_nii_with_labels.csv')

def get_preprocessed_path(original_path):
    folder = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    new_name = f"preprocessed_{filename}"
    return os.path.join(folder, new_name)

# Apply function
df['filepath'] = df['filepath'].apply(get_preprocessed_path)

# Save new CSV
df.to_csv('adni_preprocessed_metadata.csv', index=False)
print(f"✅ Updated metadata saved to adni_preprocessed_metadata.csv with {len(df)} entries.")
