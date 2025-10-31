"""
uv run random_classes/dataset_loaders/drop.py
"""

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _extract_answer(example: Dict[str, Any]) -> str:
    candidates: List[str] = []

    def extend_from_container(container: Any) -> None:
        if container is None:
            return
        if isinstance(container, dict):
            for key in ("spans", "texts", "values", "text", "answer", "answers"):
                if key not in container:
                    continue
                value = container[key]
                if isinstance(value, list):
                    for item in value:
                        cleaned = _clean_string(item)
                        if cleaned:
                            candidates.append(cleaned)
                else:
                    cleaned = _clean_string(value)
                    if cleaned:
                        candidates.append(cleaned)
            for key in ("number", "numbers"):
                number_val = container.get(key)
                if number_val is None:
                    continue
                if isinstance(number_val, list):
                    for item in number_val:
                        cleaned = _clean_string(item)
                        if cleaned:
                            candidates.append(cleaned)
                else:
                    cleaned = _clean_string(number_val)
                    if cleaned:
                        candidates.append(cleaned)
            date_val = container.get("date")
            if isinstance(date_val, dict):
                parts = [
                    _clean_string(date_val.get("month")),
                    _clean_string(date_val.get("day")),
                    _clean_string(date_val.get("year")),
                ]
                if any(parts):
                    candidates.append("-".join(part for part in parts if part))
        elif isinstance(container, list):
            for item in container:
                cleaned = _clean_string(item)
                if cleaned:
                    candidates.append(cleaned)
        else:
            cleaned = _clean_string(container)
            if cleaned:
                candidates.append(cleaned)

    extend_from_container(example.get("answers_spans"))
    extend_from_container(example.get("answer_spans"))
    extend_from_container(example.get("answer"))
    extend_from_container(example.get("answers"))
    extend_from_container(example.get("answers_numbers"))
    extend_from_container(example.get("answer_number"))

    if not candidates:
        raise ValueError(
            f"Could not determine answer for DROP example {example.get('id')}"
        )

    return candidates[0]


def generate_drop(
    config: str = "default",
    split: str = "validation",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    dataset = load_dataset("ucinlp/drop", config, split=split)

    records: List[Dict[str, Any]] = []
    for example in dataset:
        passage = _clean_string(example.get("passage")) or ""
        question = _clean_string(example.get("question")) or ""
        answer_text = _extract_answer(example)

        prompt_lines = [f"Passage: {passage}", f"Question: {question}"]

        records.append(
            {
                "prompt": "\n".join(prompt_lines),
                "answer_text": answer_text,
                "question_id": example.get("id"),
                "passage_id": example.get("passage_id"),
            }
        )

    df = pd.DataFrame(records)

    seed = kwargs.get("seed", 42)
    if shuffle_dataset:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 10,
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=df,
        given_variables=["prompt"],
        gen_variables=["answer_text"],
        name=f"drop_{config}_{split}",
        descriptions=[
            "Answer the question based on the passage. Respond with just the answer, usually a short word or phrase, either extracted from the passage or deduced from the information in the passage.",
        ],
    )

    return dataset_wrapper.generate_many(**args)
