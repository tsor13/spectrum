#!/usr/bin/env python3
"""
Representative runs:

uv run src/spectrum/distributional_alignment/eval_distributional.py --model_name qwen/Qwen3-14B --task urn --format chat
uv run src/spectrum/distributional_alignment/eval_distributional.py --model_name google/gemma-3-1b-pt --task states --format colon
uv run src/spectrum/distributional_alignment/eval_distributional.py --model_name google/gemma-3-12b-it --task urn --format chat --log_wandb
uv run src/spectrum/distributional_alignment/eval_distributional.py --model_name tsor13/spectrum-Llama-3.1-8B-v0 --task urn --format spectrum
"""
import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from spectrum.distributional_alignment.distributional_tasks.task_registry import (
    default_data_map,
)
from spectrum.format_utils import TEMPLATE_MAPS
from spectrum.lm_tools import all_logprobs, get_logprobs, prepare_batch, score_all


def validate_distributional_task(df: pd.DataFrame):
    """Validate the dataframe."""
    if df is None:
        raise ValueError("df is required")
    if df.empty:
        raise ValueError("df is empty")
    # should contain input_text, target_outputs, and target_prob columns
    if "input_text" not in df.columns:
        raise ValueError("df must have an input_text column")
    if "target_outputs" not in df.columns:
        raise ValueError("df must have a target_outputs column")
    if "target_probs" not in df.columns:
        raise ValueError("df must have a target_probs column")
    # target_prob should be a list of floats and sum to 1
    if not isinstance(df["target_probs"].iloc[0], list):
        raise ValueError("target_probs should be a list of floats")
    if not np.isclose(np.sum(df["target_probs"].iloc[0]), 1.0):
        raise ValueError("target_probs should sum to 1")


distributional_divergence_metrics = {
    # KL(P || Q)
    "kl_divergence": lambda target_probs, obs_probs: np.sum(
        np.where(target_probs > 0, target_probs * np.log(target_probs / obs_probs), 0.0)
    ),
    # Reverse KL(Q || P)
    "reverse_kl": lambda target_probs, obs_probs: np.sum(
        np.where(obs_probs > 0, obs_probs * np.log(obs_probs / target_probs), 0.0)
    ),
    # Jensen–Shannon divergence
    "js_divergence": lambda target_probs, obs_probs: 0.5
    * (
        np.sum(
            np.where(
                target_probs > 0,
                target_probs * np.log(target_probs / ((target_probs + obs_probs) / 2)),
                0.0,
            )
        )
        + np.sum(
            np.where(
                obs_probs > 0,
                obs_probs * np.log(obs_probs / ((target_probs + obs_probs) / 2)),
                0.0,
            )
        )
    ),
    # Cross-entropy H(P,Q)
    "cross_entropy": lambda target_probs, obs_probs: -np.sum(
        np.where(target_probs > 0, target_probs * np.log(obs_probs), 0.0)
    ),
    "mode_match": lambda target_probs, obs_probs: 1
    * (np.argmax(target_probs) == np.argmax(obs_probs)),
}


def evaluate_model_on_distributional_task(
    model_name,
    task_name,
    format,
    dataset_kwargs={},
    batch_size=32,
    wandb_project=None,
    wandb_name=None,
    wandb_args=None,
    tokenizer_path=None,
    max_eval=None,
    random_seed=42,
    print_only=False,
):
    task = default_data_map[task_name]()
    df = task["df"]
    output_name = task.get("output_name", "Output")
    validate_distributional_task(df)

    # Sample dataset if max_eval is specified
    if max_eval is not None and len(df) > max_eval:
        print(
            f"Sampling {max_eval} examples from {len(df)} total examples (seed={random_seed})"
        )
        df = df.sample(n=max_eval, random_state=random_seed).reset_index(drop=True)
    else:
        print(f"Using all {len(df)} examples")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    if not print_only:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        )
    else:
        model = None

    template = TEMPLATE_MAPS[format]

    EXTRA_ARG_MAP = {
        "chat": {
            "chat_prompt": "Given the description and an input, respond with just an output.",
        },
    }

    def inputs_to_messages(row):
        description_text = row.get("description_text", None)
        input_text = row.get("input_text", None)
        messages = []
        if description_text is not None:
            messages.append({"role": "description", "content": description_text})
        if input_text is not None:
            messages.append({"role": "input", "content": input_text})
        # make sure at least one message is present
        if len(messages) == 0:
            raise ValueError("At least one message is required")
        return template["messages_to_text"](
            messages, tokenizer, start_generation=True, **EXTRA_ARG_MAP.get(format, {})
        )

    df["prompt"] = df.apply(inputs_to_messages, axis=1)

    n_print = 1
    for i in range(n_print):
        prompt = df["prompt"].iloc[i]
        print(prompt + "_")
        # also print out the target outputs
        print(f"Target outputs: {df['target_outputs'].iloc[i]}")
        # also print out the target probs
        print(f"Target distribution: {df['target_probs'].iloc[i]}")
        print()

    if print_only:
        return

    # score all
    input_texts, output_texts = [], []
    row_inds, output_inds = [], []
    for row_ind, row in df.iterrows():
        input_text = row["prompt"]
        if format == "colon":
            # replace \n\nOutput with \nOutput
            input_text = input_text.replace("\n\nOutput", "\nOutput")
            # replace Output with output_name
            input_text = input_text.replace("Output", output_name)
        for output_ind, output in enumerate(row["target_outputs"]):
            if format == "colon":
                output = " " + output
            input_texts.append(input_text)
            output_texts.append(output)
            row_inds.append(row_ind)
            output_inds.append(output_ind)

    # get top 5 logprobs for random 3 prompts
    # set random seed
    np.random.seed(random_seed)
    random_prompts = np.random.choice(input_texts, size=3, replace=False)
    batch = prepare_batch(tokenizer, random_prompts)
    logprobs = get_logprobs(batch, model, tokenizer)
    logprobs_dicts = logprobs["logprobs_dicts"]
    for i in range(len(random_prompts)):
        prompt = random_prompts[i] + "_"
        print(prompt)
        # print top 5 logprobs
        for logprob in logprobs_dicts[i].keys():
            print(f"{logprob}: {logprobs_dicts[i][logprob]}")
        print()

    results = score_all(
        input_texts, output_texts, model, tokenizer, batch_size=batch_size
    )

    df["logprobs"] = df["target_outputs"].apply(lambda x: [None] * len(x))
    # add back to df
    # loop through results, make logprobs of same shape as target_outputs
    for row_ind, output_ind, nll in zip(row_inds, output_inds, results["nll"]):
        df.at[row_ind, "logprobs"][output_ind] = -nll
    df["probs"] = df["logprobs"].apply(lambda x: np.exp(x))
    df["coverage"] = df["probs"].apply(lambda x: np.sum(x))
    df["normalized_probs"] = df["probs"].apply(lambda x: x / np.sum(x))
    print(df.head())

    def calculate_metric(target_probs, normalized_probs, metric_name, eps=1e-6):
        target_probs = np.array(target_probs) + eps
        normalized_probs = np.array(normalized_probs) + eps
        # normalize
        target_probs = target_probs / np.sum(target_probs)
        normalized_probs = normalized_probs / np.sum(normalized_probs)
        return distributional_divergence_metrics[metric_name](
            target_probs, normalized_probs
        )

    # calculate distributional divergence statistics
    for metric in distributional_divergence_metrics:
        df[metric] = df.apply(
            lambda x: calculate_metric(
                x["target_probs"], x["normalized_probs"], metric
            ),
            axis=1,
        )
    print(df.head(3))
    print(df.tail(3))

    # get summary statistics - keep distributional divergence statistics (mean/std/se), coverage (mean/std/se), and length of length of target_outputs (mean/std/se)
    stat_cols = [metric for metric in distributional_divergence_metrics] + ["coverage"]
    df["n_targets"] = df["target_outputs"].apply(len)
    stat_cols += ["n_targets"]
    summary_statistics = {
        col: {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "se": df[col].sem(),
        }
        for col in stat_cols
    }
    summary_statistics["total_n"] = len(df)

    print(summary_statistics)

    if wandb_project is not None:
        wandb_config = wandb_args or {}
        wandb_config["model_name"] = model_name
        wandb_config["format"] = format
        wandb_config["task"] = task_name
        wandb.init(
            project=wandb_project,
            entity="tsor1313",
            name=wandb_name,
            config=wandb_config,
        )
        wandb.log(summary_statistics)

        # log df as a table / artifact
        print(df.head())
        wandb.log({"all_results": wandb.Table(dataframe=df)})

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_args", type=str, default=None)
    # log wandb (default false)
    parser.add_argument("--log_wandb", action="store_true", default=False)
    parser.add_argument(
        "--max_eval",
        type=int,
        default=1000,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument("--print_only", action="store_true", default=False)

    args = parser.parse_args()

    if args.log_wandb:
        if args.wandb_project is None:
            args.wandb_project = args.task + "_distributional_alignment"
        if args.wandb_name is None:
            args.wandb_name = f"{args.model_name}_{args.format}"
        if args.wandb_args is None:
            args.wandb_args = {}

    evaluate_model_on_distributional_task(
        args.model_name,
        args.task,
        args.format,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_args=args.wandb_args,
        max_eval=args.max_eval,
        random_seed=args.random_seed,
        print_only=args.print_only,
    )
