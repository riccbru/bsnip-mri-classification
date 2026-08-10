import os
import csv

# Update this with your RCC path
base_dir = "/project/aereditato/abhat/ADNI1_Complete_3Yr_3T/ADNI/ADNI1_Complete_3Yr_3T/ADNI"

output_csv = "adni_matched_nii_with_labels.csv"

# Dummy function (replace this with actual metadata matching logic if needed)
def get_label_from_path(path):
    # Placeholder: assign dummy labels based on folder name for now
    if "MCI" in path:
        return 0
    elif "AD" in path:
        return 1
    else:
        return 2  # CN

entries = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            full_path = os.path.join(root, file)
            label = get_label_from_path(full_path)
            entries.append((full_path, label))

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label"])
    writer.writerows(entries)

print(f"✅ Saved {len(entries)} entries to {output_csv}")
