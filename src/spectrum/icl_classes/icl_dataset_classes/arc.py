"""
uv run random_classes/dataset_loaders/arc.py
"""

import os
import sys
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate


def _build_prompt(stem: str, labels: List[str], texts: List[str]) -> str:
    lines = [f"Question: {stem}"]
    for label, choice_text in zip(labels, texts):
        clean_label = label.strip()
        clean_text = str(choice_text).strip()
        lines.append(f"{clean_label}. {clean_text}")
    return "\n".join(lines)


def _extract_stem(example: Dict[str, Any]) -> str:
    question_field = example.get("question")
    if isinstance(question_field, dict):
        stem = question_field.get("stem", "")
        return str(stem).strip()
    if question_field is not None:
        return str(question_field).strip()
    raise KeyError("ARC example is missing a 'question' field")


def _extract_choices(example: Dict[str, Any]) -> Dict[str, List[str]]:
    choices = example.get("choices")
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels")
        texts = choices.get("text") or choices.get("texts")
        if labels is None or texts is None:
            raise KeyError("ARC example choices missing 'label' or 'text' entries")
        return {"labels": list(labels), "texts": list(texts)}

    # Some older versions flatten choices into parallel fields like choiceA, choiceB, etc.
    derived_labels: List[str] = []
    derived_texts: List[str] = []
    for key, value in example.items():
        if key.lower().startswith("choice") and len(key) > 6:
            label = key[-1].upper()
            derived_labels.append(label)
            derived_texts.append(str(value))
    if derived_labels and derived_texts:
        return {"labels": derived_labels, "texts": derived_texts}

    raise KeyError("ARC example has unexpected choice format")


def generate_arc(
    config: str = "ARC-Challenge",
    split: str = "test",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    dataset = load_dataset("allenai/ai2_arc", config, split=split)

    records: List[Dict[str, str]] = []
    for example in dataset:
        stem = _extract_stem(example)
        choice_info = _extract_choices(example)
        labels = choice_info["labels"]
        texts = choice_info["texts"]
        if len(labels) != len(texts):
            raise ValueError(
                f"Mismatch between number of ARC choice labels {len(labels)} and texts {len(texts)} for example {example.get('id')}"
            )
        answer_raw = example.get("answerKey")
        if answer_raw is None:
            raise KeyError(f"ARC example {example.get('id')} missing 'answerKey'")
        answer_key = str(answer_raw).strip()

        prompt = _build_prompt(stem, labels, texts)
        normalized_labels = [label.strip() for label in labels]
        if answer_key not in normalized_labels:
            raise ValueError(
                f"Answer key '{answer_key}' not found among labels {normalized_labels} for ARC example {example.get('id')}"
            )

        records.append(
            {
                "prompt": prompt,
                "answer_letter": answer_key,
                "question_id": example.get("id"),
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
        gen_variables=["answer_letter"],
        name=f"arc_{config.lower().replace('-', '_')}_{split}",
        descriptions=[
            "Given a question and multiple choice answers, return the correct answer letter.",
            "Respond with only the uppercase letter corresponding to the correct option.",
        ],
    )

    return dataset_wrapper.generate_many(**args)
