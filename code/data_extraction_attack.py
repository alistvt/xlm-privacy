import json
import os
import numpy as np
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def training_data_extraction(args, tokenizer, model, val_dataset):
    """
    Randomly select examples from the train dataset, pick the first
    `args.prompt_length` tokens as the prompt, then compare the next
    tokens with the model generation. We'll measure exact match for
    different evaluation lengths dynamically.
    """
        
    def measure_metrics(reference_tokens, candidate_tokens):
        """Compute exact match, BLEU, and approximate match (>args.bleu_thresh (.75))."""
        exact_match = (candidate_tokens == reference_tokens)
        bleu = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=SmoothingFunction().method1
        )
        approx_match = (bleu > args.bleu_thresh)
        return exact_match, bleu, approx_match

    texts = val_dataset["text"]
    eval_lengths = args.eval_lengths
    results = []

    for text in texts:
        # Tokenize entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # If the text isn't long enough for our largest comparison, skip
        max_eval_length = max(eval_lengths)
        if len(tokens) < args.prompt_length + max_eval_length:
            print("WARNING")
            continue

        # Define prompt and target sequences dynamically
        prompt_tokens = tokens[:args.prompt_length]
        target_tokens = {length: tokens[args.prompt_length : args.prompt_length + length] for length in eval_lengths}
        prompt_text = tokenizer.decode(prompt_tokens)

        # Prepare input for model
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Generate output (greedy) up to the max evaluation length
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=args.prompt_length + max_eval_length,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract the generated tokens (excluding the prompt)
        generated_tokens = output[0].tolist()[args.prompt_length:]
        
        # Compute metrics dynamically
        eval_results = {}
        for length in eval_lengths:
            gen_tokens = generated_tokens[:length]
            exact_match, bleu, approx_match = measure_metrics(target_tokens[length], gen_tokens)
            eval_results[length] = {
                "exact_match": exact_match,
                "approx_match": approx_match,
                "bleu": bleu
            }

        # Store results
        results.append({
            "prompt": prompt_text,
            "evaluations": eval_results,
            "target_text": tokenizer.decode(target_tokens[max_eval_length]),
            "generated_text": tokenizer.decode(generated_tokens[:max_eval_length])
        })

    return results


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)


def compute_statistics(results):
    """Compute statistics from the extracted training data results."""
    total_samples = len(results)
    stats = {
        "total samples": total_samples,
        "exact matches": {},
        "exact matches %": {},
        "approximate matches": {},
        "approximate matches %": {},
        "average bleu": {},
        "bleu percentiles": {}
    }

    # Extract all evaluation lengths dynamically
    eval_lengths = list(results[0]["evaluations"].keys()) if results else []

    for length in eval_lengths:
        bleu_scores = []
        exact_match_count = 0
        approx_match_count = 0

        for sample in results:
            eval_data = sample["evaluations"][length]
            bleu_scores.append(eval_data["bleu"])
            if eval_data["exact_match"]:
                exact_match_count += 1
            if eval_data["approx_match"]:
                approx_match_count += 1

        stats["exact matches"][length] = exact_match_count
        stats["exact matches %"][length] = exact_match_count / total_samples
        stats["approximate matches"][length] = approx_match_count
        stats["approximate matches %"][length] = approx_match_count / total_samples
        stats["average bleu"][length] = np.mean(bleu_scores)
        stats["bleu percentiles"][length] = {
            "25th": np.percentile(bleu_scores, 25),
            "50th (median)": np.percentile(bleu_scores, 50),
            "75th": np.percentile(bleu_scores, 75)
        }

    return stats


def save_results_and_stats(results, output_dir):
    """Save results and computed statistics to JSON files."""
    with open(os.path.join(output_dir, "dea_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    stats = compute_statistics(results)

    with open(os.path.join(output_dir, "dea_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    print(f"Statistics saved to {output_dir}")

