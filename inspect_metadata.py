# File: inspect_metadata.py

import pandas as pd

df = pd.read_csv("adni_preprocessed_metadata.csv")

print(f"✅ Rows in CSV: {len(df)}")
print(f"🧪 Sample rows:\n{df.head()}")

# Check for missing or corrupted labels
missing_labels = df['diagnosis'].isnull().sum()
unique_labels = df['diagnosis'].unique()

print(f"❓ Missing labels: {missing_labels}")
print(f"🏷️ Unique labels: {unique_labels}")
