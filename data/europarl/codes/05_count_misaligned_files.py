import os

# Path to the Europarl dataset directory
DATASET_DIR = "."  # <-- update this path

# Get list of language folders
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Find shared files across all languages
file_sets = []
for lang in lang_dirs:
    lang_path = os.path.join(DATASET_DIR, lang)
    filenames = set(f for f in os.listdir(lang_path) if f.endswith(".txt"))
    file_sets.append(filenames)
common_files = set.intersection(*file_sets)

# Track mismatches
mismatch_files = []

print("Checking for misaligned files...\n")

for filename in sorted(common_files):
    line_counts = {}
    for lang in lang_dirs:
        file_path = os.path.join(DATASET_DIR, lang, filename)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                num_lines = sum(1 for _ in f)
            line_counts[lang] = num_lines
        except FileNotFoundError:
            line_counts[lang] = "MISSING"
    
    # Check if all counts are the same
    counts = list(line_counts.values())
    if len(set(counts)) != 1:
        mismatch_files.append((filename, line_counts))
        print(f"❗ Mismatch in {filename}:")
        for lang, count in sorted(line_counts.items()):
            print(f"   [{lang}]: {count}")
        print()

print(f"\nTotal mismatched files: {len(mismatch_files)}")
