import os
import numpy as np
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from huggingface_hub import login, Repository, create_repo
from utils import (generate_branch_name, generate_load_branch_name, 
                   clear_directory, get_free_gpu_memory) 

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)

def train_model(args, train_dataset, val_dataset):
    # Load tokenizer and model
    if args.epoch_to_load != 0:
        model_repo = args.model_name
        checkpoint_revision = generate_load_branch_name(args.epoch_to_load)

        model = AutoModelForCausalLM.from_pretrained(model_repo, device_map="auto", revision=checkpoint_revision)
        tokenizer = AutoTokenizer.from_pretrained(model_repo, revision=checkpoint_revision)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Hugging Face authentication
    if args.hf_token:
        login(token=args.hf_token)

    # Repository name for Hugging Face
    repo_id = f"{args.hf_username}/{args.hf_model_name}"

    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True
    )

    output_dir = f"./{args.output_dir}/checkpoints/"

    # TrainingArguments: Save only at specified epochs
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Train up to the max epoch in list
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Save after each epoch
        logging_dir=f"./logs/{args.hf_model_name}",
        logging_steps=50,
        push_to_hub=False  # We'll push manually after saving checkpoints
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    repo = Repository(local_dir=output_dir, clone_from=repo_id)
    
    for epoch in range(args.epoch_to_load, args.train_epochs):
        trainer.train()

        # Save checkpoint only if in the specified list
        # Save model checkpoint locally
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Push to a new branch
        branch_name = generate_branch_name(epoch)
        repo.git_checkout(branch_name, create_branch_ok=True)
        repo.push_to_hub(commit_message=f"Checkpoint after epoch {epoch+1}")

        print(f"Pushing {branch_name} to Hugging Face...")
    
    return trainer, tokenizer, model
