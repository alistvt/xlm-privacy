import os
from transformers import AutoTokenizer
from tqdm import tqdm

def filter_lines_by_token_length(input_file, output_file, low_threshold, high_threshold):
    """
    Reads lines from input_file, tokenizes each line using the BLOOM tokenizer,
    and writes only those lines that exceed token_threshold to output_file.

    Parameters:
    -----------
    input_file : str
        Path to the input file.
    output_file : str
        Path to the output file where filtered lines should be saved.
    token_threshold : int
        Lines with a token count strictly greater than this threshold
        will be written to the output.
    """
    
    # Initialize BLOOM tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    # Filter lines and write to new file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in tqdm(lines):
            # Tokenize line
            tokens = tokenizer.tokenize(line.strip())
            if low_threshold <= len(tokens) <= high_threshold:
                outfile.write(line)

# Example usage:
if __name__ == "__main__":
    context = {
        "en": 64,
        "sl": 108,
        "nl": 98,
        "pl": 124,
        "cs": 116,
    }
    for lang in context.keys():
        context_len = context[lang]
        input_path = f"data\\EMEA\\raw\\{lang}.txt"
        output_path = f"data\\EMEA\\processed\\{lang}\\shorts.txt"
        # threshold = int(64 * 1.820417825659151) # Example threshold
        low_threshold = int(context_len/2)
        high_threshold = context_len
        filter_lines_by_token_length(input_path, output_path, low_threshold=low_threshold, high_threshold=high_threshold)
        print(f"Filtered lines ( {low_threshold} < token count < {high_threshold}) have been written to {output_path}.")
