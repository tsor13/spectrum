import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# ensure project root and random_classes folder are on PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "random_classes"))


def load_globaloqa_dataset():
    """Load GlobalOQA dataset from Anthropic's global opinions data."""
    # Load from HuggingFace dataset
    df = pd.read_csv(
        "hf://datasets/Anthropic/llm_global_opinions/data/global_opinions.csv"
    )

    # Parse selections string into a Python dict
    def _parse_defaultdict(s: str) -> dict:
        m = re.search(r"\{.*\}", s)
        return ast.literal_eval(m.group()) if m else {}

    df["selections"] = df["selections"].apply(_parse_defaultdict)
    df["options"] = df["options"].apply(ast.literal_eval)

    processed_data = []
    for _, row in df.iterrows():
        question = row["question"]
        options = row["options"]
        selections = row["selections"]

        # Skip rows with invalid questions
        if pd.isna(question) or not isinstance(question, str):
            continue

        # For each country in this question's data
        for country, probabilities in selections.items():
            # Create input text with question and country context
            # input_text = f"Country: {country}\n{question.strip()}"
            description_text = f"Responses from a person from this country: {country}"
            input_text = question.strip()

            # Convert options to letters (A, B, C, D, etc.)
            target_outputs = [chr(65 + i) for i in range(len(options))]  # A, B, C, D...

            # Ensure probabilities are floats and sum to 1
            target_probs = [float(prob) for prob in probabilities]

            # add the options
            input_text += "\nOptions:"
            for letter, option in enumerate(options):
                input_text += f"\n{chr(65 + letter)}. {option}"

            # Normalize probabilities to ensure they sum to 1
            prob_sum = sum(target_probs)
            if prob_sum > 0:
                target_probs = [prob / prob_sum for prob in target_probs]

            processed_data.append(
                {
                    "description_text": description_text,
                    "input_text": input_text,
                    "target_outputs": target_outputs,
                    "target_probs": target_probs,
                    "country": country,
                    "question": question,
                    "options": options,
                    "source": row.get("source", "unknown"),
                }
            )

    df_processed = pd.DataFrame(processed_data)
    print(f"Successfully loaded {len(processed_data)} examples from GlobalOQA dataset")
    return {
        "df": df_processed,
        "output_name": f"Response (letter)",
    }


if __name__ == "__main__":
    data = load_globaloqa_dataset()
    df = data["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print(f"\nOutput name: {data['output_name']}")
    print(f"\nCountries: {sorted(df['country'].unique())}")
    print(f"Number of unique questions: {df['question'].nunique()}")
    breakpoint()
