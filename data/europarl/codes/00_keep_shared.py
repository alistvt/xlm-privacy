import os
from tqdm import tqdm

# Path to the Europarl dataset directory (contains language subfolders)
DATASET_DIR = "."  # Change this to your actual path

# Get list of language folders (e.g., 'en', 'de', 'nl', ...)
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Collect sets of filenames for each language
file_sets = []
for lang in lang_dirs:
    lang_path = os.path.join(DATASET_DIR, lang)
    filenames = set(f for f in os.listdir(lang_path) if f.endswith(".txt"))
    file_sets.append(filenames)

# Find the intersection (files present in ALL language folders)
common_files = set.intersection(*file_sets)
print(f"Found {len(common_files)} common files.")

# Now remove any non-common file from all language folders
for lang in tqdm(lang_dirs):
    lang_path = os.path.join(DATASET_DIR, lang)
    for filename in tqdm(os.listdir(lang_path)):
        if filename.endswith(".txt") and filename not in common_files:
            os.remove(os.path.join(lang_path, filename))

print("Cleanup complete: only shared files retained.")
