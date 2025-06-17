import os
from transformers import AutoTokenizer

# === CONFIGURATION ===
MERGED_DIR = "."  # input dir with lang/train.txt
TOKEN_THRESHOLD = 100
EN_LANG = "en"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# Get language directories
lang_dirs = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]
assert EN_LANG in lang_dirs, f"English language folder '{EN_LANG}' not found."

# Load English lines
en_path = os.path.join(MERGED_DIR, EN_LANG, "train.txt")
with open(en_path, "r", encoding="utf-8", errors="ignore") as f_en:
    en_lines = [line.rstrip("\n") for line in f_en]

# Find indices of lines that meet the token threshold
keep_indices = []
for i, line in enumerate(en_lines):
    tokens = tokenizer.encode(line.strip(), add_special_tokens=False)
    if len(tokens) >= TOKEN_THRESHOLD:
        keep_indices.append(i)

print(f"🔍 Keeping {len(keep_indices)} lines out of {len(en_lines)} based on English token threshold.")

# Filter all languages
for lang in lang_dirs:
    input_path = os.path.join(MERGED_DIR, lang, "train.txt")
    output_path = os.path.join(MERGED_DIR, lang, "val.txt")

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in:
        lines = [line.rstrip("\n") for line in f_in]

    filtered_lines = [lines[i] for i in keep_indices if i < len(lines)]

    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in filtered_lines:
            f_out.write(line + "\n")

    print(f"✅ Wrote filtered lines to {output_path}")

print("🎉 Filtering complete!")
