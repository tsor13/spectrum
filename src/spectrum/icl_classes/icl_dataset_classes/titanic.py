import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_titanic_data():
    """Load Titanic dataset from HuggingFace."""
    # Load the Titanic dataset from mstz/titanic
    ds = load_dataset("mstz/titanic")
    # Convert to pandas DataFrame
    df = ds["train"].to_pandas()

    # Convert has_survived to string for better generation
    df["has_survived"] = df["has_survived"].astype(str)

    return df


def generate_titanic_all_variables(**kwargs) -> pd.DataFrame:
    """
    Generate all Titanic passenger variables.
    Task 1: Generate all passenger information including survival status.
    """
    df = load_titanic_data()

    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)

    # Identify all relevant columns for generation
    # Exclude internal/technical columns like passenger_id, name (too specific)
    generation_columns = []
    given_columns = []

    # Check what columns are available and select appropriate ones
    available_columns = df.columns.tolist()

    # Actual column names from mstz/titanic dataset
    potential_gen_columns = [
        "has_survived",
        "passenger_class",
        "is_male",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    # Only include columns that exist in the dataset
    generation_columns = [
        col for col in potential_gen_columns if col in available_columns
    ]

    # Generate all passenger variables
    generator = GenericMultivariate(
        df=df,
        name="titanic_all_variables",
        given_variables=given_columns,
        gen_variables=generation_columns,
        descriptions=[
            "Generate complete passenger information for the Titanic dataset, including survival status, passenger class, demographics, and travel details.",
            "Create realistic Titanic passenger profiles with all relevant attributes including whether they survived the disaster.",
            "Generate: has_survived (survival outcome), passenger_class (1st/2nd/3rd class), is_male (gender), age, sibsp (siblings/spouses aboard), parch (parents/children aboard), fare (ticket price), embarked (port of embarkation).",
        ],
    )

    return generator.generate_many(**args)


def generate_titanic_survival_prediction(**kwargs) -> pd.DataFrame:
    """
    Generate survival predictions from Titanic passenger information.
    Task 2: Given passenger information, predict survival status.
    """
    df = load_titanic_data()

    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 1 to the seed to avoid the same seed for different tasks
    args["seed"] += 1

    # Identify input columns (everything except survival)
    available_columns = df.columns.tolist()

    # Actual input column names from mstz/titanic dataset (excluding survival)
    potential_input_columns = [
        "passenger_class",
        "is_male",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    # Only include columns that exist in the dataset
    input_columns = [col for col in potential_input_columns if col in available_columns]

    # Generate survival predictions
    generator = GenericMultivariate(
        df=df,
        name="titanic_survival_prediction",
        given_variables=input_columns,
        gen_variables=["has_survived"],
        descriptions=[
            "Given passenger information including class, demographics, family size, and travel details, predict whether the passenger survived the Titanic disaster.",
            "Predict Titanic passenger survival based on socioeconomic status, demographics, and family connections.",
            "Given: passenger_class (1st/2nd/3rd class), is_male (gender), age, sibsp (siblings/spouses), parch (parents/children), fare (ticket price), embarked (port). Generate: has_survived (survival status).",
        ],
    )

    return generator.generate_many(**args)
