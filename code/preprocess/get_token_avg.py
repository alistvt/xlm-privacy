import os
import math
from transformers import AutoTokenizer
from tqdm import tqdm

scales = {
    'cs.txt': 120,
    'en.txt': 64,
    'nl.txt': 100,
    'pl.txt': 120,
    'sl.txt': 108,
}

def compute_percentiles_in_dir(directory_path, n=None):
    """
    Computes the token-length thresholds for lines (sentences) in the first N lines
    of each file in a directory using the BLOOM tokenizer.
    
    Specifically, for percent p in [10, 20, ..., 90], this script finds the value T_p
    such that p% of lines exceed T_p. (In traditional terms, T_p is the (100 - p)th percentile.)
    
    Parameters:
    -----------
    directory_path: str
        Path to the directory containing text files.
    n: int
        Number of lines to consider from each file.
    """

    # Load BLOOM tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    
    # Percentiles: for p%, we want T_p such that p% of lines are above T_p
    # => T_p is actually the (100 - p)-th percentile of the distribution.
    desired_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        scale = scales.get(filename, 0)
        total_lines = 0
        # Skip subdirectories or non-text files if needed
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Consider only the first N lines
            if n:
                lines = lines[:n]
            
            # Get token lengths for each line
            token_lengths = []
            for line in tqdm(lines):
                tokens = tokenizer.tokenize(line)
                if len(tokens) > scale:
                    total_lines += 1
            
            print(f"File: {filename}")
            print(f"  Scale: {scale}")
            print(f"  Number bigger than scale: {total_lines}")

# Example usage:
if __name__ == "__main__":
    directory_path = "data\\EMEA"
    # You can change n to process a different number of lines from each file
    compute_percentiles_in_dir(directory_path)
