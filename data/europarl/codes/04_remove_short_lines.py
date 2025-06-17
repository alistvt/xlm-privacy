import os
from tqdm import tqdm
from transformers import AutoTokenizer

# === CONFIGURATION ===
DATASET_DIR = "."          # original data
OUTPUT_DIR = "new"         # where to save filtered aligned files
TOKEN_THRESHOLD = 16                               # min tokens in English line
EN_LANG = "en"                                       # reference language for filtering

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# Get list of language folders
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Make sure output directories exist
for lang in lang_dirs:
    os.makedirs(os.path.join(OUTPUT_DIR, lang), exist_ok=True)

# Get list of common files across all languages
file_sets = []
for lang in lang_dirs:
    lang_path = os.path.join(DATASET_DIR, lang)
    filenames = set(f for f in os.listdir(lang_path) if f.endswith(".txt"))
    file_sets.append(filenames)
common_files = set.intersection(*file_sets)

print(f"Filtering {len(common_files)} shared files...")

# Process each shared file
for filename in tqdm(sorted(common_files), desc="Filtering files"):
    # Step 1: Load English file and find lines that meet the token threshold
    en_path = os.path.join(DATASET_DIR, EN_LANG, filename)
    with open(en_path, "r", encoding="utf-8", errors="ignore") as f_en:
        en_lines = [line.rstrip('\n') for line in f_en]
    
    # Indices of lines with enough tokens
    keep_indices = []
    for i, line in enumerate(en_lines):
        tokens = tokenizer.encode(line.strip(), add_special_tokens=False)
        if len(tokens) >= TOKEN_THRESHOLD:
            keep_indices.append(i)
    
    # Step 2: Apply filtering to all language versions of the file
    for lang in lang_dirs:
        input_path = os.path.join(DATASET_DIR, lang, filename)
        output_path = os.path.join(OUTPUT_DIR, lang, filename)
        
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
            lines = [line.rstrip('\n') for line in f_in]
        
        # Keep only selected lines
        filtered_lines = [lines[i] for i in keep_indices if i < len(lines)]
        
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in filtered_lines:
                f_out.write(line + "\n")

print("Filtering complete! Files saved to:", OUTPUT_DIR)
