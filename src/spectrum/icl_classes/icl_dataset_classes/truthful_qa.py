"""
uv run random_classes/dataset_loaders/truthful_qa.py
"""

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

_TARGET_MAP = {
    "mc1": "mc1_targets",
    "mc2": "mc2_targets",
}


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _extract_choice_info(example: Dict[str, Any], target_key: str) -> Dict[str, Any]:
    container = example.get(target_key)
    if isinstance(container, dict):
        labels = container.get("labels") or container.get("choices")
        scores = container.get("scores")
        if labels and scores:
            cleaned_labels = [str(label).strip() for label in labels]
            cleaned_scores = [float(score) for score in scores]
            if len(cleaned_labels) != len(cleaned_scores):
                raise ValueError(
                    f"Mismatch between number of labels ({len(cleaned_labels)}) and scores ({len(cleaned_scores)}) in TruthfulQA example {example.get('question_id', example.get('id'))}"
                )
            return {
                "options": cleaned_labels,
                "correct_index": _pick_highest_score_index(cleaned_scores),
            }

    choices_field = example.get("choices")
    label_index = example.get("label")
    if choices_field is not None and label_index is not None:
        options = [str(choice).strip() for choice in choices_field]
        try:
            correct_index = int(label_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid TruthfulQA label value '{label_index}' for example {example.get('question_id', example.get('id'))}"
            ) from exc
        if correct_index < 0 or correct_index >= len(options):
            raise ValueError(
                f"TruthfulQA label index {correct_index} out of bounds for example {example.get('question_id', example.get('id'))}"
            )
        return {
            "options": options,
            "correct_index": correct_index,
        }

    raise ValueError(
        f"Unable to extract choices for TruthfulQA example {example.get('question_id', example.get('id'))}"
    )


def _pick_highest_score_index(scores: List[float]) -> int:
    max_score = max(scores)
    candidate_indices = [idx for idx, score in enumerate(scores) if score == max_score]
    if not candidate_indices:
        raise ValueError("Unable to determine correct answer for TruthfulQA example")
    return candidate_indices[0]


_CONFIG_ALIASES = {
    "mc": "multiple_choice",
    "multiple_choice": "multiple_choice",
}


def generate_truthful_qa_mc(
    config: str = "multiple_choice",
    split: str = "validation",
    target: str = "mc2",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    target_key = _TARGET_MAP.get(target.lower())
    if target_key is None:
        raise ValueError(
            f"Unsupported TruthfulQA target '{target}'. Choose from {sorted(_TARGET_MAP)}"
        )

    hf_config = _CONFIG_ALIASES.get(config.lower(), config)

    dataset = load_dataset("EleutherAI/truthful_qa_mc", hf_config, split=split)

    records: List[Dict[str, Any]] = []
    for example in dataset:
        question = _clean_text(example.get("question")) or ""
        choice_info = _extract_choice_info(example, target_key)
        options = choice_info["options"]
        correct_index = choice_info["correct_index"]

        if len(options) > len(LETTERS):
            raise ValueError(
                "Number of answer choices exceeds supported letter mapping"
            )

        prompt_lines = [f"Question: {question}"]
        for idx, option in enumerate(options):
            prompt_lines.append(f"{LETTERS[idx]}. {option}")

        records.append(
            {
                "prompt": "\n".join(prompt_lines),
                "answer_letter": LETTERS[correct_index],
                "question_id": example.get("question_id") or example.get("id"),
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
        name=f"truthful_qa_{target}_{split}",
        descriptions=[
            "You will be given a question and several answer choices, only one of which is true. Respond with the correct answer letter."
        ],
    )

    return dataset_wrapper.generate_many(**args)


def generate_truthful_qa_binary(
    config: str = "binary",
    split: str = "validation",
    target: str = "binary",
    shuffle_dataset: bool = True,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    # target_key = _TARGET_MAP.get(target.lower())
    # if target_key is None:
    #     raise ValueError(
    #         f"Unsupported TruthfulQA target '{target}'. Choose from {sorted(_TARGET_MAP)}"
    #     )

    # hf_config = _CONFIG_ALIASES.get(config.lower(), config)

    # dataset = load_dataset("EleutherAI/truthful_qa_binary", hf_config, split=split)
    # dataset = load_dataset("EleutherAI/truthful_qa_binary", split=split)

    dataset = load_dataset("EleutherAI/truthful_qa_binary", split=split)

    records: List[Dict[str, Any]] = []
    # for example in dataset:
    for i, example in enumerate(dataset):
        question = _clean_text(example.get("question"))
        # answer = _clean_text(example.get("answer")) or ""
        # for now, just include the first one
        statement = example.get("choices")[0]
        label = example.get("label")
        answer = "True" if label == 0 else "False"
        records.append(
            {
                # "prompt": question,
                # "answer": answer,
                # "question_id": example.get("question_id") or example.get("id"),
                "question": question,
                "statement": statement,
                "answer": answer,
                "question_id": i,
            }
        )
        # also add in the converse
        statement_converse = example.get("choices")[1]
        answer_converse = "True" if label == 1 else "False"
        records.append(
            {
                "question": question,
                "statement": statement_converse,
                "answer": answer_converse,
                "question_id": i,
            }
        )

    df = pd.DataFrame(records)

    seed = kwargs.get("seed", 42)
    # if shuffle_dataset:
    #     df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 32,
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=df,
        given_variables=["question", "statement"],
        gen_variables=["answer"],
        name=f"truthful_qa_{target}_{split}",
        descriptions=[
            "You will be given a question and a statement, and you need to determine if the statement is true or false."
        ],
        # TODO - add group by for ordering of statements in the same batch by question_id
        group_by=["question_id"],
    )

    return dataset_wrapper.generate_many(**args)
