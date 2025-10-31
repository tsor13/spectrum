"""
uv run src/spectrum/distributional_alignment/distributional_tasks/numbergame.py
"""

import os
import sys
from typing import List

import numpy as np
import pandas as pd

RESPONSE_OPTIONS: List[str] = ["Yes", "No"]


def _format_number_set(raw_set: str) -> str:
    parts = [part.strip() for part in str(raw_set).split("_")]
    filtered = [part for part in parts if part]
    return ", ".join(filtered)


def load_numbergame_dataset(min_responses: int = 9, random_seed: int = 42):
    """Load Number Game data aggregated into Yes/No response distributions."""
    data_path = os.path.join("data", "numbergame", "numbergame_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Number Game data not found at {data_path}. Make sure the dataset is available."
        )

    df = pd.read_csv(data_path)

    required_columns = {"set", "target", "rating", "id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in Number Game data: {missing}")

    df = df.dropna(subset=["set", "target", "rating", "id"]).copy()
    df = df[df["rating"].isin([0, 1])]

    np.random.seed(random_seed)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    grouped = df.groupby(["set", "target"], sort=False)

    records = []
    for (number_set, target_number), group in grouped:
        total = len(group)
        if total < min_responses:
            continue

        yes_count = int(group["rating"].sum())
        no_count = total - yes_count

        yes_prob = yes_count / total
        no_prob = no_count / total

        formatted_set = _format_number_set(number_set)
        target_str = str(target_number)

        description_text = (
            "You are a randomly selected participant in a study. "
            "You will be given a set of numbers which all belong to the same set or pattern, and will be given a target number which may or may not belong to the same set or pattern. "
            "Answer Yes if you think that the target number belongs to the same set, otherwise answer No."
        )
        input_text = f"Example set: {formatted_set}\nTarget number: {target_str}"

        records.append(
            {
                "description_text": description_text,
                "input_text": input_text,
                "target_outputs": list(RESPONSE_OPTIONS),
                "target_probs": [yes_prob, no_prob],
                "total_responses": total,
                "yes_count": yes_count,
                "set_raw": number_set,
                "target_number": target_str,
            }
        )

    df_processed = pd.DataFrame(records)
    print(
        f"Successfully loaded {len(df_processed)} Number Game prompts with distributions "
        f"(min_responses={min_responses})."
    )

    return {
        "df": df_processed,
        "output_name": "Target Number Membership (Yes/No)",
    }


if __name__ == "__main__":
    data = load_numbergame_dataset()
    df_out = data["df"]

    breakpoint()
    print(df_out)
    print(data["output_name"])
