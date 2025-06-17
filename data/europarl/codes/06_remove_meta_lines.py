# remove lines starting with "<" in all files
import os

# === CONFIGURATION ===
DATASET_DIR = "."        # original data
OUTPUT_DIR = "new"          # cleaned data output

# Get language folders
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Ensure output directories exist
for lang in lang_dirs:
    os.makedirs(os.path.join(OUTPUT_DIR, lang), exist_ok=True)

# Process each language and file
for lang in lang_dirs:
    lang_path = os.path.join(DATASET_DIR, lang)
    output_lang_path = os.path.join(OUTPUT_DIR, lang)

    for filename in os.listdir(lang_path):
        if not filename.endswith(".txt"):
            continue
        
        input_path = os.path.join(lang_path, filename)
        output_path = os.path.join(output_lang_path, filename)

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
            lines = f_in.readlines()
        
        # Filter out lines that start with < (after optional whitespace)
        cleaned_lines = [line for line in lines if not line.lstrip().startswith("<")]

        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(cleaned_lines)

print("✅ Tag-removal complete. Cleaned files saved to:", OUTPUT_DIR)
