import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_netflix_data(
    data_folder: str = "data/netflix",
    sample_movies: int | None = None,
    n_users: int | None = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load Netflix rating data from combined_data files and movie titles.

    Args:
        data_folder: Path to Netflix data folder
        sample_movies: Number of movies to sample for efficiency (set to None for all)

    Returns:
        DataFrame with columns: customer_id, movie_id, rating, date, title, year
    """
    cache_file = None
    if n_users is not None:
        cache_file = os.path.join(data_folder, f"cached_n_users_{n_users}.csv")
        # check if cached_n_users.csv exists
        if os.path.exists(cache_file):
            # load the cached_n_users.csv
            cached_n_users = pd.read_csv(cache_file)
            return cached_n_users

    # Load movie titles with proper encoding
    movie_titles_path = os.path.join(data_folder, "movie_titles.csv")
    print("Loading movie titles...")
    movies_df = pd.read_csv(
        movie_titles_path,
        names=["movie_id", "year", "title"],
        encoding="latin1",  # Netflix data uses latin1 encoding
        on_bad_lines="skip",
    )

    # Sample movies if specified
    if sample_movies is not None:
        np.random.seed(42)  # For reproducible sampling
        sampled_movies = np.random.choice(
            movies_df["movie_id"].values,
            size=min(sample_movies, len(movies_df)),
            replace=False,
        )
        movies_df = movies_df[movies_df["movie_id"].isin(sampled_movies)]

    # Load rating data from combined_data files
    all_ratings = []
    # change year to int and then string
    # movies df - if nan, map to "", else, do int and then str
    movies_df["year"] = movies_df["year"].fillna("-1").astype(int).astype(str)
    # replace -1 with ""
    movies_df["year"] = movies_df["year"].replace("-1", "")

    print("Loading rating data...")
    for i in range(1, 4):  # crashes loading all 4 files, trying to only do the first 3
        print(f"Loading rating data from combined_data_{i}.txt...")
        file_path = os.path.join(data_folder, f"combined_data_{i}.txt")

        current_movie_id = None

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()

                # Movie ID line (ends with colon)
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                    # Skip if this movie isn't in our sample
                    if (
                        sample_movies is not None
                        and current_movie_id not in sampled_movies
                    ):
                        current_movie_id = None
                        continue
                # Rating line
                elif current_movie_id is not None:
                    try:
                        customer_id, rating, date = line.split(",")
                        all_ratings.append(
                            {
                                "customer_id": int(customer_id),
                                "movie_id": current_movie_id,
                                "rating": int(rating),
                                "date": date,
                            }
                        )
                    except ValueError:
                        continue  # Skip malformed lines

    ratings_df = pd.DataFrame(all_ratings)

    # Merge with movie information
    merged_df = ratings_df.merge(movies_df, on="movie_id", how="inner")

    if n_users is not None:
        # sample n_users from the merged_df
        np.random.seed(random_seed)
        # get users with at least 10 ratings
        users_with_at_least_10_ratings = (
            merged_df["customer_id"]
            .value_counts()[merged_df["customer_id"].value_counts() >= 10]
            .index
        )
        sampled_users = np.random.choice(
            users_with_at_least_10_ratings, size=n_users, replace=False
        )
        sampled_df = merged_df[merged_df["customer_id"].isin(sampled_users)]
        # save the sampled_users to a csv
        sampled_df.to_csv(cache_file, index=False)
        return sampled_df
    else:
        return merged_df


def generate_netflix_individual_ratings(**kwargs) -> pd.DataFrame:
    """
    Generate individual user movie rating predictions using IndividualMultivariate.
    Each user's rating patterns are modeled individually.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=64),
        "max_inds": 1000,  # Limit number of users for efficiency
    }
    args.update(kwargs)

    # Load Netflix data
    netflix_df = load_netflix_data()  # Sample 200 movies for efficiency

    # Filter users with at least 10 ratings for meaningful patterns
    user_counts = netflix_df["customer_id"].value_counts()
    frequent_users = user_counts[user_counts >= 10].index
    netflix_df = netflix_df[netflix_df["customer_id"].isin(frequent_users)]

    # Convert rating to string for generation
    netflix_df["rating_str"] = netflix_df["rating"].astype(str)

    individual_multivariate = IndividualMultivariate(
        df=netflix_df,
        individual_id_column="customer_id",
        given_variables=["title"],
        gen_variables=["rating_str"],
        name="netflix_individual_ratings",
        descriptions=[
            "Movie ratings (1-5 stars) for a specific user.",
            "Movie ratings for a specific user.",
        ],
    )

    data = individual_multivariate.generate_many(**args)
    return data


def generate_netflix_individual_views(**kwargs) -> pd.DataFrame:
    """
    Generate individual user movie viewing predictions using separate SingleVariableIID for each user.
    Assumes users have seen all movies they rated, and predicts movie titles they've viewed.
    """
    args = {
        "seed": 43,  # Different seed from ratings task
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=64),
        "max_inds": 1000,  # Limit number of users for efficiency
    }
    args.update(kwargs)

    # Load Netflix data
    netflix_df = load_netflix_data()

    # Filter users with at least 10 ratings for meaningful patterns
    user_counts = netflix_df["customer_id"].value_counts()
    frequent_users = user_counts[user_counts >= 10].index
    netflix_df = netflix_df[netflix_df["customer_id"].isin(frequent_users)]

    # Create separate SingleVariableIID for each user with their movie titles
    all_data = []
    unique_users = netflix_df["customer_id"].unique()

    for i, user_id in enumerate(unique_users):
        user_movies = netflix_df[netflix_df["customer_id"] == user_id]["title"].tolist()

        # Create SingleVariableIID for this user's movies
        user_dataset = SingleVariableIID(
            categories=user_movies,
            name=f"netflix_user_{user_id}_views",
            descriptions=[
                "Movie titles watched by a single user.",
                "Movies a single user has seen and rated.",
            ],
            replacement=False,  # User has seen each movie only once
        )

        # Generate data for this user with incremented seed
        user_args = args.copy()
        user_args["seed"] = args["seed"] + i

        user_data = user_dataset.generate_many(**user_args)
        all_data.append(user_data)

    # Concatenate all user data
    data = pd.concat(all_data, ignore_index=True)
    return data
