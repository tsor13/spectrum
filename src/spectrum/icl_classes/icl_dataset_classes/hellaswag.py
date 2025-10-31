"""
uv run random_classes/dataset_loaders/hellaswag.py
"""

import os
import sys
from typing import Any, List

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def _format_context(row: pd.Series) -> str:
    activity = row.get("activity_label")
    ctx = row.get("ctx")
    if ctx is None and row.get("ctx_a") is not None:
        ctx_parts: List[str] = []
        if isinstance(row.get("ctx_a"), str):
            ctx_parts.append(row["ctx_a"])
        if isinstance(row.get("ctx_b"), str):
            ctx_parts.append(row["ctx_b"])
        ctx = " ".join(part.strip() for part in ctx_parts if part)
    if ctx is None:
        ctx = ""
    lines: List[str] = []
    if activity:
        lines.append(f"Activity: {activity}")
    if ctx:
        lines.append(f"Context: {ctx}")
    return "\n".join(lines)


def generate_hellaswag(
    split: str = "validation",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    dataset = load_dataset("Rowan/hellaswag", split=split)
    df = dataset.to_pandas()

    seed = kwargs.get("seed", 42)
    if shuffle_dataset:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    def build_prompt(row: pd.Series) -> str:
        header = _format_context(row)
        prompt_lines: List[str] = []
        if header:
            prompt_lines.append(header)
        prompt_lines.append("Choose the option that best completes the description:")
        for idx, ending in enumerate(row["endings"]):
            prompt_lines.append(f"{LETTERS[idx]}. {ending}")
        return "\n".join(prompt_lines)

    df["prompt"] = df.apply(build_prompt, axis=1)
    df["answer_letter"] = df["label"].apply(lambda value: LETTERS[int(value)])

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 20,
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=df,
        given_variables=["prompt"],
        gen_variables=["answer_letter"],
        name=f"hellaswag_{split}",
        descriptions=[
            "Given a context and four possible endings, output the letter of the ending that best completes the context.",
            "Respond with the single uppercase letter for the correct choice.",
        ],
    )

    return dataset_wrapper.generate_many(**args)
