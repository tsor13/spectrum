import json
import os
import sys

import numpy as np
import pandas as pd


def load_urn_dataset(n=1000, seed=42):
    # colors
    possible_colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "brown",
        "black",
        "white",
    ]

    # set the random seed
    np.random.seed(seed)

    # generate n examples
    processed_data = []
    for i in range(n):
        # choose a random number of colors from 2-5
        num_colors = np.random.randint(2, 6)
        # choose a random number of balls from 1-10, for each color number
        num_balls = np.random.randint(1, 11, num_colors)
        # choose a random color for each ball
        colors = np.random.choice(possible_colors, num_colors, replace=False)
        # choose a random number of balls for each color
        num_balls_per_color = np.random.randint(1, num_balls + 1, num_colors)
        # templatize in the following style.
        # Description: There is an urn with the following balls: 3 red balls, 10 black balls, and 1 blue ball.
        # Input: Draw a ball at random.
        description_text = (
            f"There is an urn with the following balls shuffled together: "
        )
        ball_strings = []
        for color, num_ball in zip(colors, num_balls_per_color):
            if num_ball == 1:
                ball_strings.append(f"{num_ball} {color} ball")
            else:
                ball_strings.append(f"{num_ball} {color} balls")
        description_text += ", ".join(ball_strings[:-1]) + " and " + ball_strings[-1]
        description_text += "."
        input_text = "Draw a ball at random, and tell me the color (lowercase)."
        target_outputs = colors
        target_probs = num_balls_per_color / np.sum(num_balls_per_color)
        # target_probs to list of floats
        target_probs = [float(prob) for prob in target_probs]
        processed_data.append(
            {
                "description_text": description_text,
                "input_text": input_text,
                "target_outputs": target_outputs,
                "target_probs": target_probs,
            }
        )
    df = pd.DataFrame(processed_data)
    print(f"Successfully loaded {len(processed_data)} examples from Urn dataset")
    return {
        "df": df,
        "output_name": "Response (color)",
    }

    #     processed_data.append({
    #         'description_text': description_text,
    #         'input_text': input_text,
    #         'target_outputs': target_outputs,
    #         'target_probs': target_probs,
    #     })

    # df = pd.DataFrame(processed_data)
    # print(f"Successfully loaded {len(processed_data)} examples from MoralChoice dataset")
    # return {
    #     "df": df,
    #     "output_name": "Response (A, B)",
    # }


if __name__ == "__main__":
    dataset = load_urn_dataset()
    df = dataset["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    breakpoint()
    pass
