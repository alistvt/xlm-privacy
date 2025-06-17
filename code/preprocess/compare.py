import os
from transformers import AutoTokenizer
from tqdm import tqdm

def compute_avg_tokens_per_sentence_in_dir(directory_path):
    """
    Computes the average number of tokens for the first N lines of each file
    in a directory using the BLOOM tokenizer.
    
    Parameters:
    -----------
    directory_path: str
        Path to the directory containing text files.
    n: int
        Number of lines (sentences) to consider from each file.
    """
    
    # Load BLOOM tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        
        # Skip subdirectories or non-text files if needed
        # You can also add your own checks here
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_tokens = 0
            for line in tqdm(lines):
                # Tokenize each line as a separate sentence
                tokens = tokenizer.tokenize(line)
                total_tokens += len(tokens)


            print(f"File: {filename}")
            print(f"  Number of lines processed: {len(lines)}")
            print(f"  Total tokens: {total_tokens}\n")


# Example usage:
if __name__ == "__main__":
    # Replace 'path/to/directory' with the actual directory path
    directory_path = "data\\en-sl"
    
    # You can change n to process a different number of lines
    compute_avg_tokens_per_sentence_in_dir(directory_path)