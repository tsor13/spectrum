"""
uv run random_classes/dataset_loaders/gpqa.py
"""

import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

GPQA_FILES: Dict[str, str] = {
    "main": "gpqa_main.csv",
    "diamond": "gpqa_diamond.csv",
    "extended": "gpqa_extended.csv",
}

GPQA_ZIP_PASSWORD = "deserted-untie-orchid"


def _load_gpqa_dataframe(
    split: str = "main",
    data_dir: Optional[Union[Path, str]] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    if split not in GPQA_FILES:
        raise ValueError(
            f"Unsupported GPQA split '{split}'. Valid options: {sorted(GPQA_FILES)}"
        )

    csv_name = GPQA_FILES[split]
    # candidate_paths = [data_dir / "dataset" / csv_name, data_dir / csv_name]
    candidate_paths = [f"data/gpqa/{csv_name}", f"data/gpqa/dataset/{csv_name}"]
    # to paths
    candidate_paths = [Path(path) for path in candidate_paths]

    for candidate in candidate_paths:
        if candidate.exists():
            df = pd.read_csv(candidate)
            if "Question" not in df.columns:
                raise KeyError(f"Expected a 'Question' column in {candidate}.")
            return df

    zip_path = data_dir / "dataset.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find '{csv_name}'. Expected it in '{data_dir / 'dataset'}' or alongside '{zip_path}'."
        )

    password_bytes = (password or GPQA_ZIP_PASSWORD).encode()
    with zipfile.ZipFile(zip_path) as archive:
        internal_names = [f"dataset/{csv_name}", csv_name]
        for internal_name in internal_names:
            try:
                with archive.open(internal_name, pwd=password_bytes) as file_obj:
                    df = pd.read_csv(file_obj)
                    if "Question" not in df.columns:
                        raise KeyError(
                            f"Expected a 'Question' column in '{internal_name}' inside '{zip_path}'."
                        )
                    return df
            except KeyError:
                continue
            except RuntimeError as exc:
                raise RuntimeError(
                    "Failed to decrypt GPQA archive. Provide the correct password via the 'password' argument."
                ) from exc

    raise FileNotFoundError(
        f"File '{csv_name}' not found inside '{zip_path}'. Please extract the dataset or specify a custom data_dir."
    )


def _extract_choice_columns(df: pd.DataFrame) -> Dict[str, Any]:
    lower_name_map = {col.lower(): col for col in df.columns}

    correct_col = None
    for key in ["correct answer", "correct_answer", "answer"]:
        if key in lower_name_map:
            correct_col = lower_name_map[key]
            break
    if correct_col is None:
        raise KeyError(
            "Could not locate the 'Correct Answer' column in the GPQA dataset."
        )

    incorrect_cols = []
    for col in df.columns:
        lower = col.lower()
        if lower.startswith("incorrect answer"):
            incorrect_cols.append(col)

    if not incorrect_cols:
        raise KeyError(
            "Could not locate any 'Incorrect Answer' columns in the GPQA dataset."
        )

    def sort_key(name: str) -> int:
        parts = name.split()
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 0

    incorrect_cols.sort(key=sort_key)

    return {
        "correct": correct_col,
        "incorrect": incorrect_cols,
        "all": incorrect_cols + [correct_col],
    }


LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def generate_gpqa(
    split: str = "main",
    shuffle_choices: bool = True,
    data_dir: Optional[str] = None,
    password: Optional[str] = None,
    choice_seed: Optional[int] = None,
    inverse: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    df = _load_gpqa_dataframe(split=split, data_dir=data_dir, password=password)
    column_info = _extract_choice_columns(df)

    dataset_seed = kwargs.get("seed", 42)
    rng = np.random.default_rng(
        choice_seed if choice_seed is not None else dataset_seed
    )

    subject_candidates = ["Subject", "Domain", "Topic", "Field", "Category"]

    prompts: List[str] = []
    answers: List[str] = []

    for idx, row in df.iterrows():
        choices: List[str] = []
        correct_position = None
        for column in column_info["all"]:
            value = row[column]
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text:
                continue
            choices.append(text)
            if column == column_info["correct"]:
                correct_position = len(choices) - 1

        if correct_position is None:
            raise ValueError(f"Row {idx} missing a valid correct answer entry.")
        if len(choices) > len(LETTERS):
            raise ValueError(
                f"Row {idx} has {len(choices)} choices, but only {len(LETTERS)} letters are supported."
            )

        permutation = (
            rng.permutation(len(choices))
            if shuffle_choices
            else np.arange(len(choices))
        )

        subject_value = None
        for candidate in subject_candidates:
            if candidate in row and not pd.isna(row[candidate]):
                subject_value = str(row[candidate]).strip()
                break

        prompt_lines = []
        if subject_value:
            prompt_lines.append(f"Subject: {subject_value}")
        prompt_lines.append(f"Question: {row['Question']}")
        for letter_idx, choice_idx in enumerate(permutation):
            prompt_lines.append(f"{LETTERS[letter_idx]}. {choices[choice_idx]}")

        prompts.append("\n".join(prompt_lines))
        answers.append(LETTERS[int(np.where(permutation == correct_position)[0][0])])

    processed_df = pd.DataFrame(
        {
            "prompt": prompts,
            "answer_letter": answers,
        }
    )

    args = {
        "seed": 42,
        "n_per": 1000,
        "n_iter": 20,
    }
    args.update(kwargs)

    dataset = GenericMultivariate(
        df=processed_df,
        given_variables=["prompt"],
        gen_variables=["answer_letter"],
        name=f"gpqa_{split}",
        descriptions=[
            "You will be given a difficult question. Respond with the correct answer letter."
        ],
    )

    return dataset.generate_many(**args)
