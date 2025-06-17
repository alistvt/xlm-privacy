# I want to make a script to only keep the first 20000 lines of each of the processed.txt files and save them in train.txt


# Absolutely! Here's a simple script that:


import os

# === CONFIGURATION ===
MERGED_DIR = "."  # e.g., the dir containing lang/processed.txt
MAX_LINES = 20000

# Get language directories
lang_dirs = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]

for lang in lang_dirs:
    input_path = os.path.join(MERGED_DIR, lang, "processed.txt")
    output_path = os.path.join(MERGED_DIR, lang, "train.txt")

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
        lines = [next(f_in) for _ in range(MAX_LINES) if not f_in.closed]

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines)

    print(f"✅ {lang}/train.txt saved with {len(lines)} lines.")

print("🎯 Truncation complete!")
