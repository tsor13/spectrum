"""
uv run random_classes/dataset_loaders/winogrande.py
"""

import os
import sys
from typing import Any, List

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

LETTERS = ["A", "B"]


def _format_sentence(sentence: str) -> str:
    if sentence is None:
        return ""
    if "_" in sentence:
        return sentence.replace("_", "_____")
    return sentence


def generate_winogrande(
    config: str = "winogrande_xl",
    split: str = "validation",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    dataset = load_dataset("allenai/winogrande", config, split=split)
    df = dataset.to_pandas()

    seed = kwargs.get("seed", 42)
    if shuffle_dataset:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    prompts: List[str] = []
    answers: List[str] = []

    for _, row in df.iterrows():
        sentence = _format_sentence(str(row.get("sentence", "")))
        option1 = str(row.get("option1", ""))
        option2 = str(row.get("option2", ""))

        prompt_lines = [
            f"Sentence: {sentence}",
            "Which option correctly fills in the blank?",
            f"A. {option1}",
            f"B. {option2}",
        ]
        prompts.append("\n".join(prompt_lines))

        answer_value = row.get("answer")
        try:
            answer_index = int(str(answer_value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid answer value '{answer_value}' in Winogrande row"
            ) from exc
        if answer_index < 1 or answer_index > len(LETTERS):
            raise ValueError(
                f"Winogrande answer index {answer_index} out of range for choices {LETTERS}"
            )
        answers.append(LETTERS[answer_index - 1])

    processed_df = pd.DataFrame(
        {
            "prompt": prompts,
            "answer_letter": answers,
        }
    )

    idx_values = df.get("idx")
    if idx_values is not None:
        processed_df["idx"] = idx_values.tolist()

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 10,
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=processed_df,
        given_variables=["prompt"],
        gen_variables=["answer_letter"],
        name=f"winogrande_{config}_{split}",
        descriptions=[
            "Given a sentence with a blank and two candidate fillers, respond with the letter of the correct option.",
            "Return the single uppercase letter corresponding to the correct choice.",
        ],
    )

    return dataset_wrapper.generate_many(**args)
