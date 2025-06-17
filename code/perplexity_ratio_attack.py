import os
import torch
import json
import evaluate
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def calculate_perplexity(model, tokenizer, texts, batch_size=4, add_start_token=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    
    for start_index in tqdm(range(0, len(encoded_texts), batch_size), desc="Computing Perplexity Ratios"):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token and tokenizer.bos_token_id is not None:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat([torch.ones_like(bos_tokens_tensor, dtype=torch.int64), attn_mask], dim=1)

        with torch.no_grad():
            logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = encoded_batch[..., 1:].contiguous()
        shift_attention_mask = attn_mask[..., 1:].contiguous()

        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        loss = (loss * shift_attention_mask).sum(1) / shift_attention_mask.sum(1)
        perplexity = torch.exp(loss)

        ppls.extend(perplexity.tolist())

    return ppls


def ppx_attack_loop(tokenizer, untrained_model, trained_model, dataset):
    """
    Compute the perplexity ratio of each sample using the pretrained and trained models.
    """
    results = []
    
    perplexities_trained = calculate_perplexity(trained_model, tokenizer, dataset["text"])
    # del untrained_model
    # torch.cuda.empty_cache()
    perplexities_untrained = calculate_perplexity(untrained_model, tokenizer, dataset["text"])
    
    for perplexity_untrained, perplexity_trained, text in zip(perplexities_untrained, perplexities_trained, dataset["text"]):
        token_length = len(tokenizer.encode(text, add_special_tokens=False))

        # Compute perplexity ratio
        perplexity_ratio = perplexity_untrained / perplexity_trained if perplexity_trained > 0 else float("inf")

        # Store results
        results.append({
            "text": text,
            "token_length": token_length,
            "ppx_untrained": perplexity_untrained,
            "ppx_trained": perplexity_trained,
            "ppx_ratio": perplexity_ratio
        })

    return results


def save_results_and_stats(results, output_dir):
    """Save results and computed statistics to JSON files."""
    with open(os.path.join(output_dir, "ppx_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Perplexity stats saved to {output_dir}")
