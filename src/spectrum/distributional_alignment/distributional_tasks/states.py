import json
import os
import sys

import numpy as np
import pandas as pd


def load_states_dataset():
    # colors
    all_states = [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Hampshire",
        "New Jersey",
        "New Mexico",
        "New York",
        "North Carolina",
        "North Dakota",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "West Virginia",
        "Wisconsin",
        "Wyoming",
    ]
    # unique, sort, to list
    all_states = list(set(all_states))
    all_states.sort()
    print(len(all_states))

    processed_data = []
    processed_data.append(
        {
            # 'description_text': "Name a U.S. state uniformly at random.",
            "input_text": "Name a U.S. state uniformly at random.",
            "target_outputs": all_states,
            "target_probs": np.full(len(all_states), 1.0 / len(all_states)).tolist(),
        }
    )
    df = pd.DataFrame(processed_data)
    print(
        f"Successfully loaded {len(processed_data)} examples from States Names dataset"
    )
    return {
        "df": df,
        "output_name": "Response (state)",
    }


if __name__ == "__main__":
    dataset = load_states_dataset()
    df = dataset["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print(f"Output name: {dataset['output_name']}")
    print(f"Number of unique states: {df['target_outputs'].nunique()}")
    print(f"Number of examples: {len(df)}")
    breakpoint()
    pass
