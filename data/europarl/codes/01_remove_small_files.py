# removes files smaller than 2KB from all languages in the Europarl dataset.

import os

# Path to the Europarl dataset directory
DATASET_DIR = "."  # <-- update this path

# Get list of language folders
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Collect sets of filenames for each language
file_sets = []
for lang in lang_dirs:
    lang_path = os.path.join(DATASET_DIR, lang)
    filenames = set(f for f in os.listdir(lang_path) if f.endswith(".txt"))
    file_sets.append(filenames)

# Find shared files across all language folders
common_files = set.intersection(*file_sets)

# Loop through common files and delete the ones where any language version is ≤ 2KB
deleted_files = 0
for filename in common_files:
    too_small = False
    for lang in lang_dirs:
        filepath = os.path.join(DATASET_DIR, lang, filename)
        if os.path.getsize(filepath) <= 2048:
            too_small = True
            break
    if too_small:
        for lang in lang_dirs:
            filepath = os.path.join(DATASET_DIR, lang, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        deleted_files += 1

print(f"Removed {deleted_files} shared files that had size ≤ 2KB in any language.")
