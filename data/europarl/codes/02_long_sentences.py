import os
from tqdm import tqdm
from transformers import AutoTokenizer

# Path to Europarl dataset
DATASET_DIR = "."  # <-- change this

# Load Pythia tokenizer (choose any model size, they share the tokenizer)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# Minimum number of tokens in a line
TOKEN_THRESHOLD = 100

# Get list of language folders
lang_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# Process each language
for lang in tqdm(lang_dirs):
    print(lang)
    lang_path = os.path.join(DATASET_DIR, lang)
    long_line_count = 0

    for filename in tqdm(os.listdir(lang_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(lang_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = tokenizer.encode(line, add_special_tokens=False)
                    if len(tokens) > TOKEN_THRESHOLD:
                        long_line_count += 1

    print(f"[{lang}] Lines with more than {TOKEN_THRESHOLD} tokens: {long_line_count}")
