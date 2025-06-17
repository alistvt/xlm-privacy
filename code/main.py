import argparse
import json
import os
import time
import torch
import traceback

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import training, utils, perplexity_ratio_attack, data_extraction_attack

# Clear GPU cache
torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment: Training Data Extraction from BLOOM")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True,
                        help="Choose mode: 'train' for training, 'inference' for evaluation.")

    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="bigscience/bloom-560m",
                        help="Which BLOOM model checkpoint to use (e.g. bigscience/bloom-560m, etc.).")
    parser.add_argument("--hf_username", type=str, required=False,
                        help="Your Hugging Face username.")
    parser.add_argument("--hf_model_name", type=str, required=False,
                        help="The name of the model on Hugging Face.")
    parser.add_argument("--revision_id", type=str, required=False,
                        help="revision id of the base model")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the train text file.")
    parser.add_argument("--validation_file", type=str, required=True,
                        help="Path to the validation text file.")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Path to the evaluation text file. It is for training evaluation.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save results.")
    parser.add_argument("--bleu_thresh", type=float, default=0.75)
    parser.add_argument("--ignore_dea", action='store_true', help="skips the data extraction attack.")

    # Train mode-specific arguments
    parser.add_argument("--hf_token", type=str, required=False,
                        help="Hugging Face token (required for pushing models).")
    
    parser.add_argument("--device", type=str, required=False,
                        help="Gpu device.")

    # Inference mode-specific arguments
    parser.add_argument("--train_epochs", type=int, required=False)
    parser.add_argument("--epoch_to_load", type=int, required=False,
                        help="The checkpoint epoch to load for inference (e.g., '1', '2', '4', etc.).")

    # Extraction parameters (common for both training and inference)
    parser.add_argument("--prompt_length", type=int, default=32,
                        help="Number of tokens to use as prompt when checking extraction.")
    parser.add_argument("--eval_lengths", type=str, default="[25, 50, 25, 50]",
                        help="List of token lengths for evaluation in JSON format.")

    args = parser.parse_args()

    # Convert list-based arguments from JSON format
    args.eval_lengths = json.loads(args.eval_lengths)

    if args.mode == "inference":
        if args.revision_id:
            pass
        elif args.epoch_to_load is None:
            raise ValueError("For inference mode, you must specify --epoch_to_load.")
    else:
        args.epoch_to_load = 0

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print(f"Output directory {args.output_dir} already exists...")
        args.output_dir = os.path.join(args.output_dir, str(time.time()))
        os.makedirs(args.output_dir)
        print(f"New output directory created: {args.output_dir}")

    # Save arguments for reference
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return args


def train(args):
    """ Train the model and save checkpoints. """
    print(f"Starting training.")

    # Load dataset
    train_dataset, val_dataset, eval_dataset = utils.load_data(args)

    # Train model
    try:
        trainer, tokenizer, model = training.train_model(args, train_dataset, eval_dataset)
        print("Training completed. Model saved at Hugging Face.")
    except Exception as e:
        print("Training failed with an exception:")
        traceback.print_exc()  # This prints the full traceback
    finally:
        utils.clear_directory(args.output_dir)
    

def inference(args):
    """ Load a specific model checkpoint and run evaluation. """
    print(f"Loading model from Hugging Face: {args.hf_username}/{args.hf_model_name}")
    print(f"Using checkpoint: checkpoint-{args.epoch_to_load}-epochs")

    # Load dataset
    train_dataset, val_dataset, eval_dataset = utils.load_data(args)

    # Load trained model from Hugging Face
    model_repo = f"{args.hf_username}/{args.hf_model_name}"
    checkpoint_revision = utils.generate_load_branch_name(args.epoch_to_load)

    model = AutoModelForCausalLM.from_pretrained(model_repo, revision=checkpoint_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=checkpoint_revision)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        # tokenizer.pad_token = tokenizer.eos_token
    
    print("Running Data Extraction Attack...")
    results = data_extraction_attack.training_data_extraction(args, tokenizer, model, val_dataset)
    data_extraction_attack.save_results_and_stats(results, args.output_dir)

    print("Running Perplexity Ratio Attack...")
    results = perplexity_ratio_attack.ppx_attack_loop(
        tokenizer=tokenizer,
        untrained_model=AutoModelForCausalLM.from_pretrained(args.model_name),
        trained_model=model,
        dataset=train_dataset
    )
    perplexity_ratio_attack.save_results_and_stats(results, args.output_dir)

    print("Inference completed. Results saved.")



def inference_base(args):
    """ Load a specific model checkpoint and run evaluation. 
    for the evaluation of pythia with its base model. during training.
    """
    print(f"Loading model from Hugging Face INFERENCE BASE: {args.model_name}")
    print(f"Using checkpoint: checkpoint-{args.revision_id}")

    # Load dataset
    train_dataset, val_dataset, eval_dataset = utils.load_data(args)

    # Load trained model from Hugging Face
    model_repo = args.model_name
    checkpoint_revision = args.revision_id

    trained_model=AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=checkpoint_revision)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = trained_model.to(device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        # tokenizer.pad_token = tokenizer.eos_token
    
    if not args.ignore_dea:
        print("Running Data Extraction Attack...")
        results = data_extraction_attack.training_data_extraction(args, tokenizer, trained_model, val_dataset)
        data_extraction_attack.save_results_and_stats(results, args.output_dir)

    untrained_model = AutoModelForCausalLM.from_pretrained(model_repo, revision=checkpoint_revision)
    print("Running Perplexity Ratio Attack...")
    results = perplexity_ratio_attack.ppx_attack_loop(
        tokenizer=tokenizer,
        untrained_model=untrained_model,
        trained_model=trained_model,
        dataset=train_dataset
    )
    perplexity_ratio_attack.save_results_and_stats(results, args.output_dir)

    print("Inference completed. Results saved.")


def main():
    args = parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        if args.revision_id:
            inference_base(args)
        else:
            inference(args)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'inference'.")


if __name__ == "__main__":
    main()
