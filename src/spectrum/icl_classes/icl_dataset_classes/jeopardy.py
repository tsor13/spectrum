"""
uv run random_classes/dataset_loaders/jeopardy.py
"""

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_jeopardy_data():
    """Load Jeopardy dataset from the CSV file."""
    data_path = "data/misc/JEOPARDY_CSV.csv"

    # Load the CSV data
    df = pd.read_csv(data_path)

    # Clean up column names (remove leading spaces)
    df.columns = df.columns.str.strip()

    # Filter out Final Jeopardy and Tiebreaker rounds for more standard Q&A format
    # Focus on regular Jeopardy! and Double Jeopardy! rounds
    df = df[df["Round"].isin(["Jeopardy!", "Double Jeopardy!"])].copy()

    # Filter out questions with image data (contain "href")
    df = df[~df["Question"].str.contains("href", case=False, na=False)].copy()

    # Clean up any missing values
    df = df.dropna(subset=["Question", "Answer", "Category"])

    return df


def generate_jeopardy_question_generation(**kwargs) -> pd.DataFrame:
    """
    Generate Jeopardy questions/clues.
    Task 1: Generate Jeopardy questions (clues) without any given variables - categorical task.
    """
    df = load_jeopardy_data()

    args = {
        "seed": 42,
        "n_per": -1,  # Generate ALL samples without repeats
        "n_iter": geom_and_poisson_iter(mean=64),
        "max_total": 1_000,  # no more than 1000 samples for training
    }
    args.update(kwargs)

    # Generate Jeopardy questions (clues) only
    generator = GenericMultivariate(
        df=df,
        name="jeopardy_question_generation",
        given_variables=[],
        gen_variables=["Question"],
        descriptions=[
            "Generate Jeopardy-style questions (clues). These are factual statements that describe something, and contestants must identify what is being described.",
            "Create trivia clues in the style of Jeopardy. The clues should be informative statements about people, places, things, or concepts.",
            "Generate: Question (Jeopardy clue as a factual statement).",
        ],
    )

    return generator.generate_many(**args)


def generate_jeopardy_answer_prediction(**kwargs) -> pd.DataFrame:
    """
    Generate Jeopardy answers from question only.
    Task 2: Given only the question (clue), predict the answer.
    """
    df = load_jeopardy_data()

    args = {
        "seed": 42,
        "n_per": -1,  # Generate ALL samples without repeats
        "n_iter": geom_and_poisson_iter(mean=64),
        "max_total": 1_000,  # no more than 1000 samples for training
    }
    args.update(kwargs)
    # add 1 to the seed to avoid the same seed for different tasks
    args["seed"] += 1

    # Generate Jeopardy answers from question only
    generator = GenericMultivariate(
        df=df,
        name="jeopardy_answer_prediction",
        given_variables=["Question"],
        gen_variables=["Answer"],
        descriptions=[
            "Given a Jeopardy question (clue), predict the correct answer. The answer should be the specific person, place, thing, or concept that the clue describes.",
            "Solve Jeopardy clues by providing the answer that fits the given statement. Use only the information provided in the clue.",
            "Given: Question (Jeopardy clue statement). Generate: Answer (correct response to the clue).",
        ],
    )

    return generator.generate_many(**args)
