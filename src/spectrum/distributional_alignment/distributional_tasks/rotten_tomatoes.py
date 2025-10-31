import os
import sys

import numpy as np
import pandas as pd


def load_rotten_tomatoes_dataset():
    """Load Rotten Tomatoes dataset for movie rating prediction."""
    # Load from CSV file
    data_path = os.path.join(
        "data", "rotten_tomatoes", "rotten_tomatoes_data_1970_2024", "movie_info.csv"
    )
    df = pd.read_csv(data_path)

    processed_data = []
    demographics = ["critic", "audience"]

    for _, row in df.iterrows():
        title = row["title"]
        release_date = row["release_date"]
        critic_score = row["critic_score"]
        audience_score = row["audience_score"]

        # Process both critic and audience if scores are available
        for demo in demographics:
            score_col = f"{demo}_score"
            if pd.notna(row[score_col]) and row[score_col] != "":
                # Parse percentage score (remove % sign)
                score_str = str(row[score_col]).replace("%", "")
                if score_str.isdigit():
                    score = int(score_str)

                    # # Create demographic-specific prompt
                    # if demo == "critic":
                    #     input_text = f"You are a movie critic, and you recently watched this movie: {title}\nIf you had to pick, would you say that it was good or bad?"
                    # else:  # audience
                    #     input_text = f"You recently watched this movie: {title}\nIf you had to pick, would you say that it was good or bad?"
                    if demo == "critic":
                        description_text = f'You are a movie critic. Given a movie, you are asked to simply rate it as "Good" or "Bad".'
                    else:  # audience
                        description_text = f'You recently watched the following movies. Given a movie, you are asked to simply rate it as "Good" or "Bad".'

                    input_text = "Movie: " + title

                    # Add release date if available
                    if pd.notna(release_date) and release_date != "":
                        input_text += f"\nRelease Date: {release_date}"

                    # # Add options
                    # input_text += "\nOptions:\nGood\nBad"

                    # Convert percentage to probability distribution
                    # Score is % good, so (100-score) is % bad
                    prob_good = score / 100.0
                    prob_bad = (100 - score) / 100.0

                    target_outputs = ["Good", "Bad"]
                    target_probs = [prob_good, prob_bad]

                    # Ensure probabilities sum to 1
                    target_probs = [float(prob) for prob in target_probs]

                    processed_data.append(
                        {
                            "description_text": description_text,
                            "input_text": input_text,
                            "target_outputs": target_outputs,
                            "target_probs": target_probs,
                            "title": title,
                            "release_date": release_date,
                            "demographic": demo,
                            "score": score,
                        }
                    )

    df_processed = pd.DataFrame(processed_data)
    print(f"Successfully loaded {len(processed_data)} Rotten Tomatoes examples")
    return {
        "df": df_processed,
        "output_name": "Movie Rating (Good or Bad)",
    }


if __name__ == "__main__":
    data = load_rotten_tomatoes_dataset()
    df = data["df"]
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print(f"\nOutput name: {data['output_name']}")
    print(f"Demographics: {sorted(df['demographic'].unique())}")
    print(f"Number of unique movies: {df['title'].nunique()}")
    print(f"Score range: {df['score'].min()} - {df['score'].max()}")

    # Show sample for each demographic
    for demo in df["demographic"].unique():
        demo_sample = df[df["demographic"] == demo].iloc[0]
        print(f"\nSample {demo} input_text:")
        print(demo_sample["input_text"])
        print(f"Sample target_outputs: {demo_sample['target_outputs']}")
        print(
            f"Sample target_probs: {[f'{p:.2f}' for p in demo_sample['target_probs']]}"
        )
        print(f"Sample score: {demo_sample['score']}%")
