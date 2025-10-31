import os
import sys
from typing import Dict, List

import pandas as pd

LIKERT_OPTIONS: List[str] = [
    "Strongly Agree",
    "Agree",
    "Somewhat Agree",
    "Neutral",
    "Somewhat Disagree",
    "Disagree",
    "Strongly Disagree",
]

RAW_TO_CANON: Dict[str, str] = {
    "STRONGLY_AGREE": "Strongly Agree",
    "AGREE": "Agree",
    "SOMEWHAT_AGREE": "Somewhat Agree",
    "NEUTRAL": "Neutral",
    "SOMEWHAT_DISAGREE": "Somewhat Disagree",
    "DISAGREE": "Disagree",
    "STRONGLY_DISAGREE": "Strongly Disagree",
}


def load_habermas_dataset(min_responses: int = 6):
    """Load Habermas survey data aggregated into Likert distributions."""
    data_dir = os.path.join("data", "habermas_data")
    ratings_path = os.path.join(data_dir, "hm_all_position_statement_ratings.parquet")

    if not os.path.exists(ratings_path):
        raise FileNotFoundError(
            f"Habermas ratings file not found at {ratings_path}. "
            "Make sure the dataset is available locally."
        )

    df = pd.read_parquet(ratings_path)
    # filter to metadata.provenance is "HUMAN_CITIZEN"
    df = df[df["metadata.provenance"] == "HUMAN_CITIZEN"]
    # dedup by question.id and ratings.metadata.participant_id
    df = df.drop_duplicates(subset=["question.id", "ratings.metadata.participant_id"])
    df = df[df["worker_id"].notna()]

    required_columns = {
        "question.id",
        "question.text",
        "question.affirming_statement",
        "ratings.agreement",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns in Habermas data: {missing_columns}"
        )

    # Normalize agreement labels to canonical Likert options
    ratings = df["ratings.agreement"].astype(str)
    ratings = ratings.str.replace("['", "", regex=False).str.replace(
        "']", "", regex=False
    )
    df.loc[:, "ratings.agreement"] = ratings
    df = df[df["ratings.agreement"] != "MOCK"].copy()
    df.loc[:, "ratings.agreement"] = df["ratings.agreement"].map(RAW_TO_CANON)

    df = df.dropna(
        subset=["ratings.agreement", "question.affirming_statement", "question.text"]
    ).copy()

    LIKERT_NUMBERS = [str(i) for i in range(1, 8)]

    records = []
    # option_block = "\n".join(f"{idx}: {label}" for idx, label in enumerate(LIKERT_OPTIONS, start=1))
    option_block = ""
    for idx, (labela, labelb) in enumerate(
        zip(LIKERT_OPTIONS, LIKERT_OPTIONS[::-1]), start=1
    ):
        statement = ""
        if idx < 4:
            statement = f"{idx}: {labela} with A"
        elif idx == 4:
            statement = f"{idx}: {labela}"
        else:
            statement = f"{idx}: {labelb} with B"
        option_block += f"{statement}\n"

    grouped = df.groupby(
        [
            "question.id",
            "question.text",
            "question.affirming_statement",
            "question.negating_statement",
        ]
    )

    for (question_id, question_text, statement, negating_statement), group in grouped:
        counts = group["ratings.agreement"].value_counts()
        total = int(counts.sum())
        if total < min_responses:
            continue

        target_probs = []
        for option in LIKERT_OPTIONS:
            count = int(counts.get(option, 0))
            target_probs.append(count / total)

        # description_text = "You are a randomly selected UK resident. You will be given a question and two statements, A and B. Rate your agreement with statement A on a likert scale from 1 to 7.\nOptions:\n" + option_block
        description_text = (
            "You are a randomly selected UK resident. You will be given a question and two statements, A and B. Rate which statement you most agree with on a likert scale from 1 to 7:\n"
            + option_block.strip()
        )
        input_text = (
            f"Question: {question_text}\nA: {statement}\nB: {negating_statement}"
            # f"Statement to rate:\n{question_text}"
            # f"Statement to rate: {statement}"
            # f"Statement rated on a 7-point Likert scale:\n{statement}\n\n"
            # f"Response options:\n{option_block}"
        )

        records.append(
            {
                "description_text": description_text,
                "input_text": input_text,
                "target_outputs": LIKERT_NUMBERS,
                "target_probs": target_probs,
                "question_id": question_id,
                "total_responses": total,
            }
        )

    df_processed = pd.DataFrame(records)
    print(
        f"Successfully loaded {len(df_processed)} Habermas prompts with distributions (min_responses={min_responses})."
    )

    return {
        "df": df_processed,
        "output_name": "Answer (Likert 1-7)",
    }


if __name__ == "__main__":
    data = load_habermas_dataset()
    df_out = data["df"]
    print(df_out)
    print(data["output_name"])
