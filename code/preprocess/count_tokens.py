import os
from transformers import AutoTokenizer

# Load the Pythia tokenizer (adjust size if needed)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")

# Root directory where the subdirectories are located
root_dir = "data\europarl\processed"

# Files to look for
target_filenames = {"train.txt", "val.txt"}

# Store results
results = []

for subdir, _, files in os.walk(root_dir):
    for filename in files:
        if filename in target_filenames:
            file_path = os.path.join(subdir, filename)

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                text = "".join(lines)

            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(token_ids)
            line_count = len(lines)

            # Store result
            results.append({
                "file_path": file_path,
                "lines": line_count,
                "tokens": token_count
            })

# Print results
for result in results:
    print(f"{result['file_path']}: {result['lines']} lines, {result['tokens']} tokens")
