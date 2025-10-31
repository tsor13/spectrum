"""
uv run src/spectrum/diverse_valid/eval_diverse_valid.py --model_name google/gemma-3-1b-it --task "car_make_model" --template "chat" --num_generations 100
"""

import argparse
from typing import Any, Dict, List

import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectrum.diverse_valid.generation_task import GenerationTask, InclusionTask
from spectrum.diverse_valid.task_registry import tasks
from spectrum.format_utils import TEMPLATE_MAPS, model_params


def format_with_stop_strings(
    gen_str: str, stop_strings: List[str], to_remove: List[str] | None = None
) -> str:
    # remove <pad>
    if to_remove is None:
        to_remove = ["<pad>", "<|endoftext|>"]
    for to_remove_str in to_remove:
        gen_str = gen_str.replace(to_remove_str, "")
        gen_str = gen_str.strip()
        for stop_string in stop_strings:
            # check if ends with stop_string
            # check if in gen_str
            if stop_string in gen_str:
                gen_str = gen_str[: gen_str.index(stop_string)]
                break
    # strip trailing whitespace
    gen_str = gen_str.strip()
    return gen_str


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    gen_str: str,
    gen_args: Dict[str, Any],
    gen_batch_size: int,
    num_generations: int,
) -> List[str]:
    """Generate up to num_generations samples using batched decoding."""
    generations: List[str] = []
    print(gen_str)
    for start in tqdm.tqdm(
        range(0, num_generations, gen_batch_size), desc="Generating batches"
    ):
        current_batch_size = min(gen_batch_size, num_generations - start)
        batch_strs = [gen_str] * current_batch_size
        inputs = tokenizer(batch_strs, return_tensors="pt").to(model.device)
        batch_generations = model.generate(
            **inputs,
            **gen_args,
        )
        gens = tokenizer.batch_decode(batch_generations[:, inputs.input_ids.shape[1] :])
        gens = [format_with_stop_strings(gen, gen_args["stop_strings"]) for gen in gens]
        generations.extend(gens[:current_batch_size])
        if start == 0:
            print(gens[:current_batch_size])
        if len(generations) >= num_generations:
            break
    return generations[:num_generations]


def evaluate_model_on_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: GenerationTask,
    num_generations: int,
    template: str,
    gen_batch_size: int = 16,
    include_description: bool = True,
    include_examples: bool = True,
) -> Dict[str, Any]:
    template_map = TEMPLATE_MAPS[template]
    # generate generations
    messages = task.get_messages(
        include_description=include_description,
        include_examples=include_examples,
    )
    extra_arg = {}
    if "chat" in template:
        extra_arg["chat_prompt"] = (
            "Respond with JUST the requested output, nothing else."
        )
        extra_arg["description_lambda"] = lambda x: x
        extra_arg["all_examples_in_prompt"] = True

    gen_str = template_map["messages_to_text"](
        messages=messages,
        tokenizer=tokenizer,
        start_generation=True,
        **extra_arg,
    )
    # check if tokenizer has "messages_to_text" function
    if hasattr(tokenizer, "messages_to_text"):
        print("Using messages_to_text function in tokenizer")
        gen_str = tokenizer.messages_to_text(messages=messages, start_generation=True)

    end_strings = [
        "<|eot_id|>",
        "<end_of_turn>",
        "<|im_end|>",
        "<|end_of_text|>",
        tokenizer.eos_token,
        "\n",
    ]

    gen_args = {
        "max_new_tokens": task.max_new_tokens,
        "temperature": 1,
        "top_p": 1,
        "do_sample": True,
        "stop_strings": end_strings,
        "tokenizer": tokenizer,
    }

    generations = generate_batch(
        model,
        tokenizer,
        gen_str,
        gen_args,
        gen_batch_size,
        num_generations,
    )
    # print the first 5 generations
    print(generations[:5])

    # get validity
    validity = task.validate_texts(generations)

    # get diversity here(?)
    raw_unique_texts = set(generations)
    raw_unique_pct = len(raw_unique_texts) / len(generations) if generations else 0.0
    valid_texts = [gen for gen, valid in zip(generations, validity) if valid]
    # for now, just get the number of unique texts as a percentage of the total number of generations
    unique_gens = set(valid_texts)
    unique_valid_count = len(unique_gens)
    usable_unique_pct = unique_valid_count / len(generations) if generations else 0.0
    unique_gens_pct = unique_valid_count / len(valid_texts) if valid_texts else 0.0

    # calculate pairwise collision metrics for valid texts
    n_valid = len(valid_texts)
    if n_valid <= 1:
        possible_collisions = 0
        actual_collisions = 0
        pairwise_collision = 0.0
        pairwise_uniqueness = 1.0
        pairwise_collision_sem = 0.0
    else:
        possible_collisions = (n_valid * (n_valid - 1)) // 2
        total_pairs = 0
        # count actual collisions by comparing all pairs
        actual_collisions = 0
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                total_pairs += 1
                if valid_texts[i] == valid_texts[j]:
                    actual_collisions += 1

        pairwise_collision = (
            actual_collisions / possible_collisions if possible_collisions > 0 else 0.0
        )
        pairwise_uniqueness = 1.0 - pairwise_collision

        # calculate standard error: sqrt(p*(1-p)) / sqrt(n) where n is possible_collisions
        if possible_collisions > 0:
            import math

            p = pairwise_collision
            pairwise_collision_sem = math.sqrt(p * (1 - p)) / math.sqrt(
                possible_collisions
            )
        else:
            pairwise_collision_sem = 0.0

    # TODO - maybe add some other diversity metrics, including eg ones that are irrespective of validity(?)

    coverage_pct = None
    # if task is InclusionTask, get coverage % as well
    if isinstance(task, InclusionTask):
        coverage_pct = len(set(valid_texts)) / len(task.valid_strings)

    return {
        "generations": generations,
        "validity": validity,
        "percent_valid": sum(validity) / len(validity),
        "unique_gens_pct": unique_gens_pct,
        "usable_unique_pct": usable_unique_pct,
        "unique_valid_count": unique_valid_count,
        "coverage_pct": coverage_pct,
        "raw_unique_pct": raw_unique_pct,
        "raw_unique_count": len(raw_unique_texts),
        "pairwise_collision": pairwise_collision,
        "pairwise_uniqueness": pairwise_uniqueness,
        "possible_collisions": possible_collisions,
        "actual_collisions": actual_collisions,
        "pairwise_collision_sem": pairwise_collision_sem,
        "prompt_components": {
            "description": include_description,
            "examples": include_examples,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="Model path (for manual loading)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional path to a different tokenizer. If not specified, uses the model's tokenizer.",
    )
    # task name (required)
    parser.add_argument("--task", type=str, required=True)
    # batch size
    parser.add_argument("--batch_size", type=int, default=8)
    # add wandb project
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=100)
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        choices=["spectrum", "chat", "colon", "simple"],
        help="Template to use for generation. One of: spectrum, chat, colon, simple",
    )
    parser.add_argument(
        "--prompt_components",
        type=str,
        choices=["both", "description", "examples"],
        default="both",
        help="Which task context to include when formatting prompts",
    )

    args = parser.parse_args()

    # Validation: either manual model_name or auto_kfold must be specified
    if args.model_name is None:
        raise ValueError("model_name must be specified")

    task_name = args.task
    if task_name not in tasks:
        raise ValueError(f"Task {task_name} not found in task registry")
    task = tasks[task_name]

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # load task
    task = tasks[args.task]()

    include_description = args.prompt_components in ("both", "description")
    include_examples = args.prompt_components in ("both", "examples")

    # evaluate
    results = evaluate_model_on_task(
        model=model,
        tokenizer=tokenizer,
        task=task,
        num_generations=args.num_generations,
        template=args.template,
        gen_batch_size=args.batch_size,
        include_description=include_description,
        include_examples=include_examples,
    )
    print(results)
