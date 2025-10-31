import json
import os
import sys

import numpy as np
import pandas as pd


def load_nytimes_dataset():
    """Load NYTimes reading likelihood dataset based on demographics."""
    # Load from local JSON file
    data_path = os.path.join(
        "data",
        "benchmarking-distributional-alignment",
        "nytimes",
        "NYTIMES_proportions.json",
    )
    with open(data_path, "r") as f:
        data = json.load(f)

    processed_data = []
    demographics = ["Democrat", "Republican", "Male", "Female"]

    for book_title, book_data in data.items():
        genre = book_data["genre"]
        summary = book_data["summary"]
        options = book_data["MC_options"]

        # For each demographic group
        for demographic in demographics:
            if demographic in book_data:
                # Get the response counts for this demographic
                response_counts = book_data[demographic]

                # Create input text with book info and demographic context
                # input_text = f"Respondent demographic: {demographic}\nHow likely are you to read this book?\nBook: {book_title}\nGenre: {genre}\nSummary: {summary}"
                description_text = f"You are a random survey respondent. Respondent demographic: {demographic}"
                input_text = f"How likely are you to read this book?\nBook: {book_title}\nGenre: {genre}\nSummary: {summary}"

                # Remove any existing "Answer:" lines
                input_text = "\n".join(
                    [
                        line
                        for line in input_text.split("\n")
                        if line.strip() != "Answer:"
                    ]
                )

                # Add the options to the input text
                input_text += "\nOptions:"
                for option in options:
                    input_text += f"\n{option}"

                # Use the first character of each option as target outputs (1, 2, 3, 4)
                target_outputs = [option.split(":")[0] for option in options]

                # Convert counts to probabilities
                total_responses = sum(response_counts.values())
                if total_responses > 0:
                    # Map options to probabilities in the correct order
                    target_probs = []
                    for option in options:
                        count = response_counts.get(option, 0)
                        target_probs.append(count / total_responses)
                else:
                    # Uniform distribution if no responses
                    target_probs = [1.0 / len(options)] * len(options)

                # Ensure probabilities are floats and sum to 1
                target_probs = [float(prob) for prob in target_probs]

                processed_data.append(
                    {
                        "description_text": description_text,
                        "input_text": input_text,
                        "target_outputs": target_outputs,
                        "target_probs": target_probs,
                        "book_title": book_title,
                        "genre": genre,
                        "summary": summary,
                        "demographic": demographic,
                        "total_responses": total_responses,
                    }
                )

    df = pd.DataFrame(processed_data)
    print(f"Successfully loaded {len(processed_data)} examples from NYTimes dataset")
    return {
        "df": df,
        "output_name": "Reading Likelihood (1, 2, 3, 4)",
    }


if __name__ == "__main__":
    data = load_nytimes_dataset()
    df = data["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print(f"\nOutput name: {data['output_name']}")
    print(f"\nDemographics: {sorted(df['demographic'].unique())}")
    print(f"Number of unique books: {df['book_title'].nunique()}")
    print(f"Genres: {sorted(df['genre'].unique())}")

    # Show a sample input_text
    print(f"\nSample input_text:")
    print(df.iloc[0]["input_text"])
    print(f"\nSample target_outputs: {df.iloc[0]['target_outputs']}")
    print(f"Sample target_probs: {df.iloc[0]['target_probs']}")
