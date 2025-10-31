"""
Representative launch commands (adjust ports as needed when using accelerate):

uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum
uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum --ignore_missing_datasets
uv run -- accelerate launch --config_file launch_configs/accelerate_config_2.yaml --num_processes 2 --gradient_accumulation_steps 2 --main_process_port 6001 src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum --debug --per_device_train_batch_size 1
uv run -- accelerate launch --config_file launch_configs/accelerate_config.yaml --num_processes 4 --gradient_accumulation_steps 2 --main_process_port 6002 src/spectrum/train.py --model_name meta-llama/Llama-3.1-8B-Instruct --format spectrum --debug --per_device_train_batch_size 1
uv run src/spectrum/train.py --model_name Qwen/Qwen3-0.6B --format spectrum --gradient_accumulation_steps 2
"""

import os
import sys

import argparse
import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, SchedulerType, TrainerCallback
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import pad

import wandb
from spectrum.icl_classes.data_cache import (
    get_safe_model_name,
    load_tokenized_datasets,
)
from spectrum.icl_classes.dataloaders import train_defaults, val_defaults
from spectrum.icl_classes.utils import format_loss_texts
from spectrum.sample_utils import MyDataCollatorForLanguageModeling

version = "v0"


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    embedding_params = 0
    attention_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

            # Categorize trainable parameters
            if any(
                embed_name in name
                for embed_name in [
                    "embed_tokens",
                    "wte",
                    "embeddings.word_embeddings",
                    "lm_head",
                    "embed_out",
                ]
            ):
                embedding_params += param.numel()
            elif any(
                attn_name in name
                for attn_name in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ):
                attention_params += param.numel()
            else:
                other_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

    if trainable_params > 0:
        print(
            f"  └─ embedding LoRA: {embedding_params:,} ({100 * embedding_params / trainable_params:.1f}%)"
        )
        print(
            f"  └─ attention LoRA: {attention_params:,} ({100 * attention_params / trainable_params:.1f}%)"
        )
        if other_params > 0:
            print(
                f"  └─ other params: {other_params:,} ({100 * other_params / trainable_params:.1f}%)"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SFT model with TRL.")
    parser.add_argument(
        "--model_name", type=str, help="Name or path of the model to fine-tune."
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=f"models_{version}",
        help="Root directory for saving outputs.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=1, help="Number of steps between logging."
    )
    # eval steps
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1024,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--save_steps", type=int, default=8, help="Save model checkpoint every N steps."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        help="Enable bf16 mixed precision training.",
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable bf16 mixed precision training.",
    )
    # add debug flag
    parser.add_argument(
        "--debug", dest="debug", action="store_true", help="Enable debug mode."
    )
    # Template format argument
    parser.add_argument(
        "--format",
        dest="format",
        type=str,
        default="spectrum",
        choices=["spectrum", "colon", "chat"],
        help="Template format to use. 'spectrum' uses spectrum format.",
    )
    # add eval_before_train flag
    parser.add_argument(
        "--eval_before_train",
        dest="eval_before_train",
        action="store_true",
        help="Evaluate before training.",
    )
    parser.add_argument(
        "--no_eval_before_train",
        dest="eval_before_train",
        action="store_false",
        help="Skip evaluation before training.",
    )
    parser.set_defaults(eval_before_train=True)
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="If set, ignore cache and re-tokenize datasets.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help=f"wandb project name (default: spectrum_training_{version} or spectrum_training_debug_{version} if debug)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for training.",
    )
    parser.add_argument(
        "--constant_lr",
        action="store_true",
        help="If set, use constant learning rate scheduler. Otherwise, use trainer's default (linear).",
    )
    # LoRA configuration arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA (Low-Rank Adaptation) training.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank parameter. Higher values = more parameters but potentially better performance.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=80,
        help="LoRA alpha parameter for scaling. Set to 80 (5x rank) to make SFT learning rates work well for LoRA.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate. Set to 0.0 for maximum adaptation capacity.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of target modules for LoRA. Common choices: q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--lora_embed_tokens",
        action="store_true",
        default=True,
        help="Add LoRA to input embedding layer (embed_tokens). Useful for training specific token representations.",
    )
    parser.add_argument(
        "--no_lora_embed_tokens",
        dest="lora_embed_tokens",
        action="store_false",
        help="Disable LoRA on input embedding layer.",
    )
    parser.add_argument(
        "--lora_lm_head",
        action="store_true",
        default=True,
        help="Add LoRA to output embedding layer (lm_head). Useful for training specific token generation.",
    )
    parser.add_argument(
        "--no_lora_lm_head",
        dest="lora_lm_head",
        action="store_false",
        help="Disable LoRA on output embedding layer.",
    )
    # add loss_all_tokens flag
    parser.add_argument(
        "--loss_all_tokens",
        action="store_true",
        default=False,
        help="If set, loss on all tokens.",
    )
    # add ablate only last as flag
    parser.add_argument(
        "--ablate_only_last",
        action="store_true",
        default=False,
        help="If set, only calculate loss on the last in-context example.",
    )
    parser.add_argument(
        "--ablate_only_first",
        action="store_true",
        default=False,
        help="If set, only calculate loss on the first in-context example.",
    )
    # add ignore_missing_datasets
    parser.add_argument(
        "--ignore_missing_datasets",
        action="store_true",
        default=False,
        help="If set, ignore missing datasets.",
    )
    return parser.parse_args()


args = parse_args()


model_name = args.model_name
safe_model_name = get_safe_model_name(model_name)
trl_save_dir = os.path.join(args.output_root, safe_model_name)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# add pad token id if not present
if tokenizer.pad_token is None:
    print("No pad token found, setting to eos token")
    tokenizer.pad_token = tokenizer.eos_token



# Create load_kwargs dict to include tokenizer if needed
load_kwargs = {
    "force_reload": args.force_reload,
    "model_name": model_name,
    "save": True,
    "format": args.format,
    # Keep individual flags for backward compatibility with data loading
    "max_length": args.max_length,
    "include_prior_prob": 0.8,
    "ignore_missing_datasets": args.ignore_missing_datasets,
}

# Make sure tokenizer is available for all template formatters
load_kwargs["tokenizer"] = tokenizer

print("Using default train/test split")
print(f"Train datasets: {train_defaults}")
print(f"Val datasets: {val_defaults}")

if not args.debug:
    train_dataset = load_tokenized_datasets(train_defaults, **load_kwargs)
    val_dataset = load_tokenized_datasets(val_defaults, **load_kwargs)
else:
    train_dataset = load_tokenized_datasets(train_defaults, **load_kwargs)
    val_dataset = load_tokenized_datasets(val_defaults, **load_kwargs)

# if debug, truncate only to first 64 examples from each
if args.debug:
    # shuffle train deterministically first
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.select(range(64))
    val_dataset = val_dataset.select(range(64))

print("Train dataset length: ", len(train_dataset))
print("Val dataset length: ", len(val_dataset))


project_name = (
    args.wandb_project
    if args.wandb_project is not None
    else f"spectrum_training_{version}"
)
if args.wandb_project is None and args.debug:
    project_name = f"spectrum_training_debug_{version}"

wandb_name = f"{safe_model_name}"
# add LoRA to name
if args.use_lora:
    lora_name = f"_lora_r{args.lora_r}"
    if args.lora_embed_tokens or args.lora_lm_head:
        embed_parts = []
        if args.lora_embed_tokens:
            embed_parts.append("embed")
        if args.lora_lm_head:
            embed_parts.append("lmhead")
        lora_name += f"_{'_'.join(embed_parts)}"
    wandb_name = f"{wandb_name}{lora_name}"
# add template format to name
wandb_name = f"{wandb_name}_{args.format}"

wandb.init(
    project=project_name,
    entity="tsor1313",  # <-- your W&B username/org
    name=wandb_name,
)

model_init_kwargs = {}
if "gemma" in model_name:
    model_init_kwargs["attn_implementation"] = "eager"

# LoRA configuration
peft_config = None
if args.use_lora:
    target_modules = [module.strip() for module in args.lora_target_modules.split(",")]

    # Add embedding layers if enabled
    embedding_modules = []
    if args.lora_embed_tokens:
        # Common names for input embedding layers across different architectures
        embedding_modules.extend(["embed_tokens", "wte", "embeddings.word_embeddings"])
    if args.lora_lm_head:
        # Common names for output embedding layers
        embedding_modules.extend(["lm_head", "embed_out"])

    if embedding_modules:
        target_modules.extend(embedding_modules)
        print(f"Adding embedding layers to LoRA targets: {embedding_modules}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(
        f"LoRA enabled with config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    print(f"Target modules: {target_modules}")
else:
    print("LoRA disabled - using full fine-tuning")

model_dir = os.path.join(trl_save_dir, "models")
if args.use_lora:
    lora_suffix = f"_lora_r{args.lora_r}"
    if args.lora_embed_tokens or args.lora_lm_head:
        embed_parts = []
        if args.lora_embed_tokens:
            embed_parts.append("embed")
        if args.lora_lm_head:
            embed_parts.append("lmhead")
        lora_suffix += f"_{'_'.join(embed_parts)}"
    model_dir = os.path.join(model_dir, lora_suffix)
model_dir = os.path.join(model_dir, f"_{args.format}")
print(model_dir)
# make model dir if it doesn't exist
os.makedirs(model_dir, exist_ok=True)


# Enhanced data collator that tracks token counts
class TokenTrackingDataCollator(MyDataCollatorForLanguageModeling):
    def torch_call(self, examples):
        # Call parent method to get the batch
        output = super().torch_call(examples)

        # Count non -100 tokens in this batch
        if "labels" in output:
            labels = output["labels"]
            non_masked_count = (labels != -100).sum().item()

            # Add to appropriate total (we'll determine train/eval context from trainer state)
            if token_counters["is_training"]:
                token_counters["total_train_non_masked"] += non_masked_count
            else:
                token_counters["total_eval_non_masked"] += non_masked_count

        return output


class TokenCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        token_counters["is_training"] = True

    def on_evaluate(self, args, state, control, **kwargs):
        token_counters["is_training"] = False
        # Reset eval counter
        token_counters["total_eval_non_masked"] = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if token_counters["is_training"]:
                logs["train/non_masked_tokens_total"] = token_counters[
                    "total_train_non_masked"
                ]
            else:
                logs["eval/non_masked_tokens_total"] = token_counters[
                    "total_eval_non_masked"
                ]

        # Also log directly to wandb to ensure it gets there
        if token_counters["is_training"]:
            wandb.log(
                {
                    "train/non_masked_tokens_total": token_counters[
                        "total_train_non_masked"
                    ]
                }
            )
        else:
            wandb.log(
                {
                    "eval/non_masked_tokens_total": token_counters[
                        "total_eval_non_masked"
                    ]
                }
            )


# --- NEW: Weight precision logger callback ---
class WeightPrecisionLoggerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            dtypes = set(p.dtype for p in model.parameters())
            print(
                f"[WeightPrecisionLogger] Model parameter dtypes at train start: {dtypes}"
            )
            if "wandb" in sys.modules:
                import wandb

                wandb.log({"weight_precision_train_start": str(dtypes)})

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None:
            dtypes = set(p.dtype for p in model.parameters())
            print(f"[WeightPrecisionLogger] Model parameter dtypes at eval: {dtypes}")
            if "wandb" in sys.modules:
                import wandb

                wandb.log({"weight_precision_eval": str(dtypes)})


# Global counters for token tracking
token_counters = {
    "total_train_non_masked": 0,
    "total_eval_non_masked": 0,
    "is_training": True,
}


# Create token counting callback
token_callback = TokenCountCallback()
# Create weight precision logger callback
weight_precision_logger = WeightPrecisionLoggerCallback()

# Build training_args_dict, conditionally set lr_scheduler_type
training_args_dict = {
    "max_length": args.max_length,
    "output_dir": model_dir,
    "logging_dir": os.path.join(trl_save_dir, "logs"),
    "logging_steps": args.logging_steps,
    "per_device_train_batch_size": args.per_device_train_batch_size,
    "per_device_eval_batch_size": args.per_device_train_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "num_train_epochs": args.num_train_epochs,
    "save_strategy": "steps",
    "save_total_limit": 1,
    "save_steps": args.save_steps,
    "report_to": "wandb",
    "eval_strategy": "steps",
    "eval_steps": args.eval_steps,
    "bf16": args.bf16,
    "model_init_kwargs": model_init_kwargs,
    "learning_rate": args.learning_rate,
}
if args.constant_lr:
    training_args_dict["lr_scheduler_type"] = SchedulerType.CONSTANT

training_args = SFTConfig(**training_args_dict)


def filter_only_last(example):
    # only keep the last contiguous sequence of non -100 tokens
    labels = example["labels"]

    # Find all non -100 positions
    non_masked_positions = [i for i, label in enumerate(labels) if label != -100]

    if not non_masked_positions:
        # If no non -100 tokens, return as is
        example["labels"] = labels
        return example

    # Find the last contiguous sequence by working backwards
    last_pos = non_masked_positions[-1]
    start_of_last_sequence = last_pos

    # Work backwards through the non-masked positions to find the start of the last contiguous block
    for i in range(len(non_masked_positions) - 1, 0, -1):
        current_pos = non_masked_positions[i]
        prev_pos = non_masked_positions[i - 1]

        # If there's a gap (difference > 1), then the contiguous sequence starts at current_pos
        if current_pos - prev_pos > 1:
            start_of_last_sequence = current_pos
            break
        else:
            # They're contiguous, so extend the sequence backwards
            start_of_last_sequence = prev_pos

    # Create new labels with everything masked except the last contiguous sequence
    new_labels = [-100] * len(labels)
    for i in range(start_of_last_sequence, last_pos + 1):
        if labels[i] != -100:
            new_labels[i] = labels[i]

    example["labels"] = new_labels
    return example


def filter_only_first(example):
    # only keep the first contiguous sequence of non -100 tokens
    labels = example["labels"]

    # Find all non -100 positions
    non_masked_positions = [i for i, label in enumerate(labels) if label != -100]

    if not non_masked_positions:
        # If no non -100 tokens, return as is
        example["labels"] = labels
        return example

    # Find the first contiguous sequence by working forwards
    first_pos = non_masked_positions[0]
    end_of_first_sequence = first_pos

    # Work forwards through the non-masked positions to find the end of the first contiguous block
    for i in range(len(non_masked_positions) - 1):
        current_pos = non_masked_positions[i]
        next_pos = non_masked_positions[i + 1]

        # If there's a gap (difference > 1), then the contiguous sequence ends at current_pos
        if next_pos - current_pos > 1:
            end_of_first_sequence = current_pos
            break
        else:
            # They're contiguous, so extend the sequence forwards
            end_of_first_sequence = next_pos

    # Create new labels with everything masked except the first contiguous sequence
    new_labels = [-100] * len(labels)
    for i in range(first_pos, end_of_first_sequence + 1):
        if labels[i] != -100:
            new_labels[i] = labels[i]

    example["labels"] = new_labels
    return example


# ablation for first or last example only
if args.ablate_only_last:
    # map each example to the last contiguous sequence
    train_dataset = train_dataset.map(lambda x: filter_only_last(x), batched=False)
if args.ablate_only_first:
    # map each example to the first contiguous sequence
    train_dataset = train_dataset.map(lambda x: filter_only_first(x), batched=False)


# log some data instances
# randomly select 30 data instances from train and log them to wandb after formatting
train_data_sample = train_dataset.select(random.sample(range(len(train_dataset)), 30))

# Option 1: Log as HTML table for better visualization
html_table_rows = []
for i, text in enumerate(train_data_sample["loss_texts"]):
    formatted_html = format_loss_texts(text[:20])[:4000]
    html_table_rows.append([i, formatted_html])

wandb.log(
    {
        "train_data_sample_table": wandb.Table(
            columns=["sample_id", "formatted_text"], data=html_table_rows
        )
    }
)

# Option 2: Also log as individual HTML artifacts for detailed viewing
for i, loss_texts in enumerate(
    train_data_sample["loss_texts"][:10]
):  # Just first 5 for artifacts
    formatted_html = format_loss_texts(loss_texts[:20])[:4000]
    wandb.log({f"train_sample_{i}": wandb.Html(formatted_html)})

# print the input_ids for the first example
print(train_data_sample[0]["input_ids"])


data_collator = TokenTrackingDataCollator(
    pad_token_id=tokenizer.pad_token_id,
    completion_only_loss=True,
    pad_to_multiple_of=None,
)

if args.loss_all_tokens:
    print("Loss on all tokens")
    # map the input
    data_collator.completion_only_loss = False


if "gemma" in model_name:
    # Hacky fix for Zero-3 init with Gemma-3 - didn't need this for training originally, but something broke in an update to transformers / trl / accelerate / deepspeed / etc.
    # Suggested by claude - seems to work fine!
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

    _original_init = PreTrainedModel._init_weights

    def _zero3_safe_init(self, module):
        """Skip padding_idx zeroing for partitioned embeddings."""
        if isinstance(module, nn.Embedding):
            # Check if parameter is ZeRO-3 partitioned
            if hasattr(module.weight, "ds_id"):
                # Skip the problematic padding_idx zeroing
                # It's not critical for training convergence
                return

        _original_init(self, module)

    PreTrainedModel._init_weights = _zero3_safe_init


trainer = SFTTrainer(
    model=model_name,
    # model=model,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    callbacks=[token_callback, weight_precision_logger],
    peft_config=peft_config,
)
trainer._signature_columns = ["input_ids", "attention_mask", "labels"]

# Print trainable parameters with debug info
if args.use_lora:
    print("=== LoRA Debug Info ===")
    print(f"PEFT config: {peft_config}")
    print(f"Model type: {type(trainer.model)}")

    # Check which target modules actually exist in the model
    print("\n=== Target Module Validation ===")
    all_module_names = set(name for name, _ in trainer.model.named_modules())
    target_modules = peft_config.target_modules

    found_modules = []
    missing_modules = []

    for target in target_modules:
        # Check if target matches any module (could be partial match)
        matches = [name for name in all_module_names if target in name]
        if matches:
            found_modules.extend(matches)
            print(f"✓ '{target}' found: {len(matches)} matches")
        else:
            missing_modules.append(target)
            print(f"✗ '{target}' NOT FOUND in model")

    # Special validation for embedding layers
    print("\n=== Embedding Layer Validation ===")
    embed_layer_candidates = [
        "embed_tokens",
        "wte",
        "embeddings.word_embeddings",
        "lm_head",
        "embed_out",
    ]

    for embed_name in embed_layer_candidates:
        if embed_name in target_modules:
            matches = [name for name in all_module_names if embed_name in name]
            if matches:
                print(f"✓ Embedding layer '{embed_name}' found: {matches}")
            else:
                print(
                    f"✗ WARNING: Embedding layer '{embed_name}' specified but not found!"
                )

    # Show actual embedding layer names for this model
    print(f"\n=== Actual Embedding Layers in {type(trainer.model).__name__} ===")
    try:
        input_embed = trainer.model.get_input_embeddings()
        output_embed = trainer.model.get_output_embeddings()

        # Find the names of these layers
        input_embed_name = None
        output_embed_name = None

        for name, module in trainer.model.named_modules():
            if module is input_embed:
                input_embed_name = name
            if module is output_embed:
                output_embed_name = name

        print(
            f"Input embedding layer: {input_embed_name} ({type(input_embed).__name__})"
        )
        print(
            f"Output embedding layer: {output_embed_name} ({type(output_embed).__name__})"
        )

        # Check if these are in our target modules
        embedding_targets = [t for t in target_modules if t in embed_layer_candidates]
        if embedding_targets:
            if input_embed_name and not any(
                t in input_embed_name for t in embedding_targets
            ):
                print(
                    f"⚠️  WARNING: Input embedding '{input_embed_name}' may not match targets {embedding_targets}"
                )
            if output_embed_name and not any(
                t in output_embed_name for t in embedding_targets
            ):
                print(
                    f"⚠️  WARNING: Output embedding '{output_embed_name}' may not match targets {embedding_targets}"
                )

    except Exception as e:
        print(f"Could not identify embedding layers: {e}")

    print("\n=== PEFT Adapter Check ===")
    if hasattr(trainer.model, "peft_config"):
        print(f"PEFT config found: {trainer.model.peft_config}")
    else:
        print("WARNING: No PEFT config found on model!")

    if missing_modules:
        print(
            f"\n❌ CRITICAL: {len(missing_modules)} target modules not found: {missing_modules}"
        )
        print(
            "This may indicate architecture mismatch. Check target_modules for this model type."
        )
    else:
        print(f"\n✅ All {len(target_modules)} target modules found successfully")

    print_trainable_parameters(trainer.model)

# evaluate and print results
if not args.debug and args.eval_before_train:
    results = trainer.evaluate()
    print(results)

print("Training...")

trainer.train()
trainer.save_model()
print("Training complete")
