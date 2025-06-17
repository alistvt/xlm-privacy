import argparse
import json
import os
import random
import shutil
import numpy as np
import torch
import subprocess

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_free_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
    )
    memory_free = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return memory_free  # list of free memory in MiB for each GPU


def load_data(args):
    """
    Loads data from text files (one line = one sample).
    Returns two Dataset objects: train_dataset, val_dataset.
    """
    # Read train text file line-by-line
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    
    # Read validation text file line-by-line
    with open(args.validation_file, 'r', encoding='utf-8') as f:
        val_lines = [line.strip() for line in f if line.strip()]
        
    # Read eval text file line-by-line
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        eval_lines = [line.strip() for line in f if line.strip()]
    
    # Create Dataset objects from these lines
    # Each line is an entry in the "text" column
    train_dataset = Dataset.from_dict({"text": train_lines})
    val_dataset = Dataset.from_dict({"text": val_lines})
    eval_dataset = Dataset.from_dict({"text": eval_lines})
    
    return train_dataset, val_dataset, eval_dataset


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)


def generate_branch_name(epoch):
    return f"epoch-{epoch+1}"


def generate_load_branch_name(epoch):
    return f"epoch-{epoch}"


def clear_directory(dir_path):
    """
    Removes all files and subdirectories from `dir_path`,
    """
    # Recursively delete the entire directory
    shutil.rmtree(dir_path, ignore_errors=True)
