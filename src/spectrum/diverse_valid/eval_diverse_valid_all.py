"""
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --model_name google/gemma-3-1b-it --template chat --num_generations 100 --prompt_components description
"""

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Core imports
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

# Import existing functions from single-task script
from spectrum.diverse_valid.eval_diverse_valid import (
    evaluate_model_on_task,
    format_with_stop_strings,
    generate_batch,
)
from spectrum.diverse_valid.generation_task import (
    FunctionTask,
    GenerationTask,
    InclusionTask,
)
from spectrum.diverse_valid.task_registry import tasks
from spectrum.format_utils import TEMPLATE_MAPS


def clean_model_name_for_wandb(model_name: str) -> str:
    """Clean model name to be suitable for wandb project names."""
    # Remove path separators and special characters
    clean_name = model_name.replace("/", "_").replace("\\", "_")
    clean_name = clean_name.replace("-", "_").replace(".", "_")
    # Remove common prefixes
    if clean_name.startswith("models_"):
        clean_name = clean_name[7:]
    return clean_name


def evaluate_all_tasks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    template: str,
    num_generations: int,
    batch_size: int,
    skip_tasks: Optional[List[str]] = None,
    include_description: bool = True,
    include_examples: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model on all registered tasks.

    Args:
        model: The language model
        tokenizer: The tokenizer
        template: Template type (spectrum, chat, colon, simple)
        num_generations: Number of generations per task
        batch_size: Generation batch size
        skip_tasks: List of task names to skip

    Returns:
        Dictionary containing all results and aggregate metrics
    """
    if skip_tasks is None:
        skip_tasks = []

    print(f"\n=== Evaluating model on ALL tasks ===")
    print(f"Model: {model.config._name_or_path}")
    print(f"Template: {template}")
    print(f"Generations per task: {num_generations}")
    print(f"Total tasks available: {len(tasks)}")
    print(f"Tasks to skip: {skip_tasks}")

    successful_results = {}
    failed_tasks = {}
    task_timings = {}

    # Get list of tasks to evaluate
    tasks_to_evaluate = [
        (name, factory) for name, factory in tasks.items() if name not in skip_tasks
    ]

    print(f"Tasks to evaluate: {len(tasks_to_evaluate)}")

    # Evaluate each task
    for task_name, task_factory in tqdm(tasks_to_evaluate, desc="Evaluating tasks"):
        print(f"\n--- Evaluating task: {task_name} ---")

        start_time = time.time()
        try:
            # Create task instance
            task = task_factory()

            # Run evaluation
            result = evaluate_model_on_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                num_generations=num_generations,
                template=template,
                gen_batch_size=batch_size,
                include_description=include_description,
                include_examples=include_examples,
            )

            # Add task metadata to result
            result["task_metadata"] = {
                "name": task_name,
                "description": (
                    task.description if hasattr(task, "description") else None
                ),
                "max_new_tokens": task.max_new_tokens,
                "task_type": (
                    "InclusionTask"
                    if isinstance(task, InclusionTask)
                    else "FunctionTask"
                ),
                "prompt_components": {
                    "description": include_description,
                    "examples": include_examples,
                },
            }

            # Store generation prompt for analysis
            # We'll need to reconstruct this from the task
            messages = task.get_messages(
                include_description=include_description,
                include_examples=include_examples,
                # all_examples_in_prompt=True,
            )
            # from spectrum.chat_utils import TEMPLATE_MAPS
            template_map = TEMPLATE_MAPS[template]

            # Apply same message transformation as in evaluate_model_on_task
            if template == "chat":
                user_message = "Generate."
                new_messages = []
                has_input = False
                for message in messages:
                    if message["role"] == "description":
                        new_messages.append(
                            {
                                "role": "system",
                                "content": "Generate something that fits this description. Don't generate anything else, just the desired generation output.\nDescription: "
                                + message["content"],
                            }
                        )
                    elif message["role"] == "input":
                        has_input = True
                        new_messages.append(
                            {"role": "user", "content": message["content"]}
                        )
                    elif message["role"] == "output":
                        if not has_input:
                            new_messages.append(
                                {"role": "user", "content": user_message}
                            )
                        new_messages.append(
                            {"role": "assistant", "content": message["content"]}
                        )
                if len(messages) == 1:
                    new_messages.append({"role": "user", "content": user_message})
                messages = new_messages

            gen_str = template_map["messages_to_text"](
                messages=messages,
                tokenizer=tokenizer,
                start_generation=True,
                # start_gen_as=template_map.get("start_gen_as", "output"),
            )

            if template == "chat" and "Qwen3" in model.config._name_or_path:
                gen_str = gen_str + "<think>Okay, let's generate.</think>"

            result["generation_prompt"] = gen_str

            successful_results[task_name] = result

            end_time = time.time()
            task_timings[task_name] = end_time - start_time

            raw_unique_pct = result.get("raw_unique_pct", 0.0)
            unique_valid_count = result.get("unique_valid_count", 0)
            print(
                f"✅ {task_name}: {result['percent_valid']:.1%} valid, "
                f"{result['unique_gens_pct']:.1%} unique, "
                f"{unique_valid_count} unique valids, "
                f"{result['usable_unique_pct']:.1%} usable unique, "
                f"{raw_unique_pct:.1%} raw unique, "
                f"{result.get('pairwise_collision', 0.0):.3f} pairwise collision"
            )

        except Exception as e:
            end_time = time.time()
            task_timings[task_name] = end_time - start_time
            failed_tasks[task_name] = str(e)
            print(f"❌ {task_name}: Failed with error: {e}")
            breakpoint()
            continue

    # Calculate aggregate metrics
    if successful_results:
        valid_percentages = [
            result["percent_valid"] for result in successful_results.values()
        ]
        unique_percentages = [
            result["unique_gens_pct"] for result in successful_results.values()
        ]
        usable_unique_percentages = [
            result["usable_unique_pct"] for result in successful_results.values()
        ]
        unique_valid_counts = [
            result.get("unique_valid_count", 0)
            for result in successful_results.values()
        ]
        raw_unique_percentages = [
            result.get("raw_unique_pct", 0.0) for result in successful_results.values()
        ]
        coverage_percentages = [
            result["coverage_pct"]
            for result in successful_results.values()
            if result["coverage_pct"] is not None
        ]
        pairwise_collision_rates = [
            result.get("pairwise_collision", 0.0)
            for result in successful_results.values()
        ]
        pairwise_uniqueness_rates = [
            result.get("pairwise_uniqueness", 1.0)
            for result in successful_results.values()
        ]

        def compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {"mean": None, "std": None, "sem": None, "count": 0}
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_val = variance**0.5
            sem_val = std_val / (len(values) ** 0.5) if len(values) > 0 else None
            return {
                "mean": mean_val,
                "std": std_val,
                "sem": sem_val,
                "count": len(values),
            }

        percent_stats = compute_stats(valid_percentages)
        unique_stats = compute_stats(unique_percentages)
        usable_unique_stats = compute_stats(usable_unique_percentages)
        unique_valid_count_stats = compute_stats(unique_valid_counts)
        coverage_stats = compute_stats(coverage_percentages)
        raw_unique_stats = compute_stats(raw_unique_percentages)
        pairwise_collision_stats = compute_stats(pairwise_collision_rates)
        pairwise_uniqueness_stats = compute_stats(pairwise_uniqueness_rates)

        aggregate_metrics = {
            "mean_percent_valid": percent_stats["mean"],
            "std_percent_valid": percent_stats["std"],
            "sem_percent_valid": percent_stats["sem"],
            "mean_unique_gens_pct": unique_stats["mean"],
            "std_unique_gens_pct": unique_stats["std"],
            "sem_unique_gens_pct": unique_stats["sem"],
            "mean_usable_unique_pct": usable_unique_stats["mean"],
            "std_usable_unique_pct": usable_unique_stats["std"],
            "sem_usable_unique_pct": usable_unique_stats["sem"],
            "mean_unique_valid_count": unique_valid_count_stats["mean"],
            "std_unique_valid_count": unique_valid_count_stats["std"],
            "sem_unique_valid_count": unique_valid_count_stats["sem"],
            "mean_coverage_pct": coverage_stats["mean"],
            "std_coverage_pct": coverage_stats["std"],
            "sem_coverage_pct": coverage_stats["sem"],
            "mean_raw_unique_pct": raw_unique_stats["mean"],
            "std_raw_unique_pct": raw_unique_stats["std"],
            "sem_raw_unique_pct": raw_unique_stats["sem"],
            "mean_pairwise_collision": pairwise_collision_stats["mean"],
            "std_pairwise_collision": pairwise_collision_stats["std"],
            "sem_pairwise_collision": pairwise_collision_stats["sem"],
            "mean_pairwise_uniqueness": pairwise_uniqueness_stats["mean"],
            "std_pairwise_uniqueness": pairwise_uniqueness_stats["std"],
            "sem_pairwise_uniqueness": pairwise_uniqueness_stats["sem"],
            "count_percent_valid": percent_stats["count"],
            "count_unique_gens_pct": unique_stats["count"],
            "count_usable_unique_pct": usable_unique_stats["count"],
            "count_unique_valid_count": unique_valid_count_stats["count"],
            "count_coverage_pct": coverage_stats["count"],
            "count_raw_unique_pct": raw_unique_stats["count"],
            "count_pairwise_collision": pairwise_collision_stats["count"],
            "count_pairwise_uniqueness": pairwise_uniqueness_stats["count"],
        }
    else:
        aggregate_metrics = {
            "mean_percent_valid": 0.0,
            "std_percent_valid": 0.0,
            "sem_percent_valid": 0.0,
            "mean_unique_gens_pct": 0.0,
            "std_unique_gens_pct": 0.0,
            "sem_unique_gens_pct": 0.0,
            "mean_coverage_pct": None,
            "std_coverage_pct": None,
            "sem_coverage_pct": None,
            "mean_usable_unique_pct": 0.0,
            "std_usable_unique_pct": 0.0,
            "sem_usable_unique_pct": 0.0,
            "mean_unique_valid_count": 0.0,
            "std_unique_valid_count": 0.0,
            "sem_unique_valid_count": 0.0,
            "mean_raw_unique_pct": 0.0,
            "std_raw_unique_pct": 0.0,
            "sem_raw_unique_pct": 0.0,
            "mean_pairwise_collision": 0.0,
            "std_pairwise_collision": 0.0,
            "sem_pairwise_collision": 0.0,
            "mean_pairwise_uniqueness": 1.0,
            "std_pairwise_uniqueness": 0.0,
            "sem_pairwise_uniqueness": 0.0,
            "count_percent_valid": 0,
            "count_unique_gens_pct": 0,
            "count_usable_unique_pct": 0,
            "count_unique_valid_count": 0,
            "count_coverage_pct": 0,
            "count_raw_unique_pct": 0,
            "count_pairwise_collision": 0,
            "count_pairwise_uniqueness": 0,
        }

    # Compile final results
    results = {
        "model_name": model.config._name_or_path,
        "template": template,
        "num_generations": num_generations,
        "prompt_components": {
            "description": include_description,
            "examples": include_examples,
        },
        "total_tasks_attempted": len(tasks_to_evaluate),
        "successful_tasks": len(successful_results),
        "failed_tasks": list(failed_tasks.keys()),
        "aggregate_metrics": aggregate_metrics,
        "task_results": successful_results,
        "failed_task_errors": failed_tasks,
        "task_timings": task_timings,
        "evaluation_timestamp": time.time(),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model.config._name_or_path}")
    print(f"Template: {template}")
    print(f"Successful tasks: {len(successful_results)}/{len(tasks_to_evaluate)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    if aggregate_metrics["mean_percent_valid"]:
        print(
            f"Mean validity: {aggregate_metrics['mean_percent_valid']:.1%} ± {aggregate_metrics.get('sem_percent_valid', 0.0):.1%}"
        )
        print(
            f"Mean uniqueness: {aggregate_metrics['mean_unique_gens_pct']:.1%} ± {aggregate_metrics.get('sem_unique_gens_pct', 0.0):.1%}"
        )
        print(
            f"Mean unique valids: {aggregate_metrics['mean_unique_valid_count']:.2f} ± {aggregate_metrics.get('sem_unique_valid_count', 0.0):.2f}"
        )
        if aggregate_metrics.get("mean_usable_unique_pct") is not None:
            print(
                f"Mean usable unique: {aggregate_metrics['mean_usable_unique_pct']:.1%} ± {aggregate_metrics.get('sem_usable_unique_pct', 0.0):.1%}"
            )
        if aggregate_metrics.get("mean_raw_unique_pct") is not None:
            print(
                f"Mean raw unique: {aggregate_metrics['mean_raw_unique_pct']:.1%} ± {aggregate_metrics.get('sem_raw_unique_pct', 0.0):.1%}"
            )
        if aggregate_metrics.get("mean_pairwise_collision") is not None:
            print(
                f"Mean pairwise collision: {aggregate_metrics['mean_pairwise_collision']:.3f} ± {aggregate_metrics.get('sem_pairwise_collision', 0.0):.3f}"
            )
        if aggregate_metrics.get("mean_pairwise_uniqueness") is not None:
            print(
                f"Mean pairwise uniqueness: {aggregate_metrics['mean_pairwise_uniqueness']:.3f} ± {aggregate_metrics.get('sem_pairwise_uniqueness', 0.0):.3f}"
            )
        if aggregate_metrics["mean_coverage_pct"]:
            print(
                f"Mean coverage: {aggregate_metrics['mean_coverage_pct']:.1%} ± {aggregate_metrics.get('sem_coverage_pct', 0.0):.1%}"
            )

    if failed_tasks:
        print(f"\nFailed tasks: {list(failed_tasks.keys())}")

    return results


def initialize_wandb_run(
    model_name: str,
    template: str,
    num_generations: int,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    skip_tasks: Optional[List[str]] = None,
    prompt_components: Optional[Dict[str, bool]] = None,
) -> Optional[object]:
    """Initialize wandb run with appropriate config."""
    try:
        wandb.login()

        # Set default project name
        if wandb_project is None:
            clean_name = clean_model_name_for_wandb(model_name)
            wandb_project = f"diverse_valid_{clean_name}"

        # Set default run name
        if wandb_run_name is None:
            model_short = model_name.split("/")[-1] if "/" in model_name else model_name
            wandb_run_name = f"{model_short}_{template}_all_tasks"

        wandb_run = wandb.init(
            project=wandb_project,
            entity="tsor1313",
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "template": template,
                "num_generations": num_generations,
                "total_available_tasks": len(tasks),
                "skip_tasks": skip_tasks or [],
                "evaluation_type": "all_tasks",
                "prompt_components": prompt_components or {},
            },
        )

        print(f"Initialized wandb run: {wandb_project}/{wandb_run_name}")
        return wandb_run

    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None


def log_aggregate_metrics(results: Dict[str, Any]) -> None:
    """Log aggregate metrics to wandb."""
    metrics = results["aggregate_metrics"]

    wandb.log(
        {
            "aggregate/percent_valid_mean": metrics["mean_percent_valid"],
            "aggregate/percent_valid_std": metrics["std_percent_valid"],
            "aggregate/percent_valid_sem": metrics.get("sem_percent_valid"),
            "aggregate/unique_gens_pct_mean": metrics["mean_unique_gens_pct"],
            "aggregate/unique_gens_pct_std": metrics["std_unique_gens_pct"],
            "aggregate/unique_gens_pct_sem": metrics.get("sem_unique_gens_pct"),
            "aggregate/unique_valid_count_mean": metrics.get("mean_unique_valid_count"),
            "aggregate/unique_valid_count_std": metrics.get("std_unique_valid_count"),
            "aggregate/unique_valid_count_sem": metrics.get("sem_unique_valid_count"),
            "aggregate/usable_unique_pct_mean": metrics.get("mean_usable_unique_pct"),
            "aggregate/usable_unique_pct_std": metrics.get("std_usable_unique_pct"),
            "aggregate/usable_unique_pct_sem": metrics.get("sem_usable_unique_pct"),
            "aggregate/raw_unique_pct_mean": metrics.get("mean_raw_unique_pct"),
            "aggregate/raw_unique_pct_std": metrics.get("std_raw_unique_pct"),
            "aggregate/raw_unique_pct_sem": metrics.get("sem_raw_unique_pct"),
            "aggregate/pairwise_collision_mean": metrics.get("mean_pairwise_collision"),
            "aggregate/pairwise_collision_std": metrics.get("std_pairwise_collision"),
            "aggregate/pairwise_collision_sem": metrics.get("sem_pairwise_collision"),
            "aggregate/pairwise_uniqueness_mean": metrics.get(
                "mean_pairwise_uniqueness"
            ),
            "aggregate/pairwise_uniqueness_std": metrics.get("std_pairwise_uniqueness"),
            "aggregate/pairwise_uniqueness_sem": metrics.get("sem_pairwise_uniqueness"),
            "aggregate/total_tasks_evaluated": results["successful_tasks"],
            "aggregate/failed_tasks_count": len(results["failed_tasks"]),
            "aggregate/success_rate": results["successful_tasks"]
            / results["total_tasks_attempted"],
        }
    )

    if metrics["mean_coverage_pct"] is not None:
        wandb.log(
            {
                "aggregate/coverage_pct_mean": metrics["mean_coverage_pct"],
                "aggregate/coverage_pct_std": metrics["std_coverage_pct"] or 0.0,
                "aggregate/coverage_pct_sem": metrics.get("sem_coverage_pct") or 0.0,
            }
        )


def create_wandb_tables(results: Dict[str, Any]) -> None:
    """Create and log wandb tables with task summary and detailed generations."""
    task_results = results["task_results"]

    # Task Summary Table
    task_summary_data = []
    detailed_generations_data = []

    for task_name, result in task_results.items():
        metadata = result["task_metadata"]

        # Task summary row
        task_summary_data.append(
            [
                task_name,
                metadata["task_type"],
                result["percent_valid"],
                result["unique_gens_pct"],
                result.get("unique_valid_count"),
                result.get("raw_unique_pct"),
                result.get("usable_unique_pct"),
                result.get("coverage_pct"),
                result.get("pairwise_collision"),
                result.get("pairwise_uniqueness"),
                result.get("possible_collisions"),
                result.get("actual_collisions"),
                len(result["generations"]),
                metadata["max_new_tokens"],
                metadata.get("description", ""),
                (
                    result["generation_prompt"][:200] + "..."
                    if len(result["generation_prompt"]) > 200
                    else result["generation_prompt"]
                ),
            ]
        )

        # Detailed generations data
        for i, (generation, is_valid) in enumerate(
            zip(result["generations"], result["validity"])
        ):
            detailed_generations_data.append(
                [
                    task_name,
                    i,
                    generation,
                    is_valid,
                    len(generation),
                    len(generation.split()) if generation else 0,
                ]
            )

    # Create and log task summary table
    task_summary_table = wandb.Table(
        columns=[
            "task_name",
            "task_type",
            "percent_valid",
            "unique_gens_pct",
            "unique_valid_count",
            "raw_unique_pct",
            "usable_unique_pct",
            "coverage_pct",
            "pairwise_collision",
            "pairwise_uniqueness",
            "possible_collisions",
            "actual_collisions",
            "num_generations",
            "max_new_tokens",
            "description",
            "generation_prompt_preview",
        ],
        data=task_summary_data,
    )
    wandb.log({"task_summary": task_summary_table})

    # Create and log detailed generations table
    detailed_table = wandb.Table(
        columns=[
            "task_name",
            "generation_index",
            "generation_text",
            "is_valid",
            "length",
            "word_count",
        ],
        data=detailed_generations_data,
    )
    wandb.log({"detailed_generations": detailed_table})


def create_wandb_artifacts(results: Dict[str, Any]) -> None:
    """Create and log wandb artifacts with complete data."""

    # Main results artifact
    artifact_main = wandb.Artifact("all_tasks_results", type="evaluation_results")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2, default=str)
        temp_path = f.name

    artifact_main.add_file(temp_path, name="all_tasks_results.json")
    wandb.log_artifact(artifact_main)
    os.unlink(temp_path)

    # Valid generations by task
    valid_generations_by_task = {}
    invalid_generations_by_task = {}
    generation_prompts_by_task = {}

    for task_name, result in results["task_results"].items():
        valid_gens = [
            gen
            for gen, valid in zip(result["generations"], result["validity"])
            if valid
        ]
        invalid_gens = [
            gen
            for gen, valid in zip(result["generations"], result["validity"])
            if not valid
        ]

        valid_generations_by_task[task_name] = valid_gens
        invalid_generations_by_task[task_name] = invalid_gens
        generation_prompts_by_task[task_name] = result["generation_prompt"]

    # Valid generations artifact
    artifact_valid = wandb.Artifact("valid_generations", type="dataset")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_generations_by_task, f, indent=2)
        temp_path = f.name
    artifact_valid.add_file(temp_path, name="valid_generations_by_task.json")
    wandb.log_artifact(artifact_valid)
    os.unlink(temp_path)

    # Invalid generations artifact
    artifact_invalid = wandb.Artifact("invalid_generations", type="dataset")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_generations_by_task, f, indent=2)
        temp_path = f.name
    artifact_invalid.add_file(temp_path, name="invalid_generations_by_task.json")
    wandb.log_artifact(artifact_invalid)
    os.unlink(temp_path)

    # Generation prompts artifact
    artifact_prompts = wandb.Artifact("generation_prompts", type="prompts")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(generation_prompts_by_task, f, indent=2)
        temp_path = f.name
    artifact_prompts.add_file(temp_path, name="generation_prompts_by_task.json")
    wandb.log_artifact(artifact_prompts)
    os.unlink(temp_path)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Multi-Task Diversity Evaluation for Text Generation"
    )

    # Model and tokenizer
    parser.add_argument("--model_name", type=str, required=True, help="Model path")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional path to different tokenizer",
    )

    # Generation parameters
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        choices=["spectrum", "chat", "colon", "simple", "chat_simple"],
        help="Template to use for generation",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=100,
        help="Number of generations per task",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for generation"
    )
    parser.add_argument(
        "--prompt_components",
        type=str,
        choices=["both", "description", "examples"],
        default="both",
        help="Which task context to include when formatting prompts",
    )

    # Task selection
    parser.add_argument(
        "--skip_tasks",
        type=str,
        default="",
        help="Comma-separated list of task names to skip",
    )

    # Wandb parameters
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name (defaults to diverse_valid_{model_name})",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to auto-generated)",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Output
    parser.add_argument(
        "--output_file", type=str, default=None, help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Parse skip_tasks
    skip_tasks = [task.strip() for task in args.skip_tasks.split(",") if task.strip()]

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, trust_remote_code=True
    )

    # Evaluate all tasks
    include_description = args.prompt_components in ("both", "description")
    include_examples = args.prompt_components in ("both", "examples")

    results = evaluate_all_tasks(
        model=model,
        tokenizer=tokenizer,
        template=args.template,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        skip_tasks=skip_tasks,
        include_description=include_description,
        include_examples=include_examples,
    )

    # Initialize wandb if requested
    wandb_run = None
    if not args.no_wandb:
        wandb_run = initialize_wandb_run(
            model_name=args.model_name,
            template=args.template,
            num_generations=args.num_generations,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            skip_tasks=skip_tasks,
            prompt_components={
                "description": include_description,
                "examples": include_examples,
            },
        )

        if wandb_run:
            try:
                # Log aggregate metrics
                log_aggregate_metrics(results)

                # Create and log tables
                create_wandb_tables(results)

                # Create and log artifacts
                create_wandb_artifacts(results)

                # Add wandb URL to results
                results["wandb_url"] = wandb_run.url

                print(f"Results logged to wandb: {wandb_run.url}")

            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

    # Save results to file if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(
        f"Successful tasks: {results['successful_tasks']}/{results['total_tasks_attempted']}"
    )
    if results["aggregate_metrics"]["mean_percent_valid"]:
        print(
            f"Overall validity: {results['aggregate_metrics']['mean_percent_valid']:.1%}"
        )
        print(
            f"Overall uniqueness: {results['aggregate_metrics']['mean_unique_gens_pct']:.1%}"
        )
        print(
            f"Overall unique valids: {results['aggregate_metrics']['mean_unique_valid_count']:.2f}"
        )

    return results


if __name__ == "__main__":
    main()
