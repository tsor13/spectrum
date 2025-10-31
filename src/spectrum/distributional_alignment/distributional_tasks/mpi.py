import os
import sys

import numpy as np
import pandas as pd


def load_mpi_dataset():
    """Load MPI (Moral Psychology Inventory) dataset for response prediction."""
    # Load response data from CSV file
    # data_path = os.path.join(project_root, "data", "mpi", "mpi_human_resp.csv")
    data_path = os.path.join("data", "mpi", "mpi_human_resp.csv")
    df = pd.read_csv(data_path)

    # Load question text from CSV file
    questions_path = os.path.join(
        "data", "mpi", "AI_Pluralistic_Alignment", "data", "mpi_questions.csv"
    )
    questions_df = pd.read_csv(questions_path, delimiter="\t")

    # Get the item columns (I1 through I120)
    item_columns = [f"I{i}" for i in range(1, 121)]

    processed_data = []

    # For each item, calculate the response distribution and create a distributional task
    for item_idx, item_col in enumerate(item_columns):
        if item_col in df.columns:
            # Get all responses for this item (excluding NaN/missing values)
            responses = df[item_col].dropna()

            # Calculate the distribution of responses (0-5 scale)
            response_counts = responses.value_counts().sort_index()

            # Ensure all possible values (0-5) are represented
            for val in range(6):
                if val not in response_counts.index:
                    response_counts[val] = 0

            response_counts = response_counts.sort_index()

            # Convert to probabilities
            total_responses = response_counts.sum()
            if total_responses > 0:
                target_probs = [
                    response_counts[val] / total_responses for val in range(6)
                ]
            else:
                # Uniform distribution if no responses
                target_probs = [1.0 / 6] * 6

            # Get the question text from questions_df
            if item_idx < len(questions_df):
                question_text = (
                    "Statement: You " + questions_df.iloc[item_idx]["text"].lower()
                )
            else:
                # question_text = f"item {item_col[1:]}. I"
                question_text = f"Statement: You {item_col[1:]}"

            # Create input text
            item_number = item_col[3:]  # Remove 'I' prefix
            # input_text = f"Rate your agreement with the following statement on a scale from 0-5, where 0 means \"strongly disagree\" and 5 means \"strongly agree.\"\n{question_text}"
            description_text = f'You are a random survey respondent. Rate your agreement with the following statement on a scale from 0-5, where 0 means "strongly disagree" and 5 means "strongly agree."'
            input_text = question_text

            # Add the options to the input text
            input_text += "\nOptions:"
            for option_val in range(6):
                input_text += f"\n{option_val}: {['Strongly disagree', 'Disagree', 'Slightly disagree', 'Slightly agree', 'Agree', 'Strongly agree'][option_val]}"

            # Target outputs are the possible ratings (0, 1, 2, 3, 4, 5)
            target_outputs = [str(option_val) for option_val in range(6)]

            # Ensure probabilities are floats and sum to 1
            target_probs = [float(prob) for prob in target_probs]

            # Calculate statistics
            mean_response = float(responses.mean()) if len(responses) > 0 else 2.5
            std_response = float(responses.std()) if len(responses) > 1 else 1.0

            processed_data.append(
                {
                    "description_text": description_text,
                    "input_text": input_text,
                    "target_outputs": target_outputs,
                    "target_probs": target_probs,
                    "item_number": item_number,
                    "total_responses": int(total_responses),
                    "mean_response": mean_response,
                    "std_response": std_response,
                }
            )

    df_processed = pd.DataFrame(processed_data)
    print(f"Successfully loaded {len(processed_data)} MPI items")
    return {
        "df": df_processed,
        "output_name": "Agreement Rating (0-5)",
    }


if __name__ == "__main__":
    data = load_mpi_dataset()
    df = data["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print(f"\nOutput name: {data['output_name']}")
    print(f"Number of MPI items: {len(df)}")
    print(
        f"Total responses range: {df['total_responses'].min()} - {df['total_responses'].max()}"
    )
    print(
        f"Mean response range: {df['mean_response'].min():.2f} - {df['mean_response'].max():.2f}"
    )

    # Show a sample input_text
    print(f"\nSample input_text:")
    print(df.iloc[0]["input_text"])
    print(f"\nSample target_outputs: {df.iloc[0]['target_outputs']}")
    print(f"Sample target_probs: {[f'{p:.3f}' for p in df.iloc[0]['target_probs']]}")
    print(f"Sample mean response: {df.iloc[0]['mean_response']:.2f}")
