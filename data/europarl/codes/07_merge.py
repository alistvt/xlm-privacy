import os

# === CONFIGURATION ===
INPUT_DIR = "."           # the directory with cleaned language folders
OUTPUT_DIR = "new"   # where to save the single output file per lang

# Get language directories
lang_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

# Ensure output directories exist
for lang in lang_dirs:
    os.makedirs(os.path.join(OUTPUT_DIR, lang), exist_ok=True)

# Process each language
for lang in lang_dirs:
    lang_input_path = os.path.join(INPUT_DIR, lang)
    lang_output_path = os.path.join(OUTPUT_DIR, lang, "processed.txt")

    with open(lang_output_path, "w", encoding="utf-8") as f_out:
        for filename in sorted(os.listdir(lang_input_path)):
            if not filename.endswith(".txt"):
                continue
            file_path = os.path.join(lang_input_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    f_out.write(line)

    print(f"✅ Concatenated files for [{lang}] -> {lang_output_path}")

print("✅ All languages processed.")
