import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_gsm8k_question(**kwargs) -> pd.DataFrame:
    """
    Generate just the question from GSM8K dataset.
    """
    # Load GSM8K dataset from HuggingFace
    ds = load_dataset("gsm8k", "main")
    # Convert to pandas DataFrame
    df = ds["train"].to_pandas()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)

    # Generate questions only
    generator = GenericMultivariate(
        df=df,
        name="gsm8k_question",
        given_variables=[],
        gen_variables=["question"],
        descriptions=[
            "Generate grade school math word problems covering arithmetic, basic algebra, and word problem reasoning.",
            "Create diverse math word problems involving real-world scenarios and numerical reasoning.",
            "Math word problems suitable for grade school students.",
        ],
    )

    return generator.generate_many(**args)


def generate_gsm8k_answer_from_question(**kwargs) -> pd.DataFrame:
    """
    Generate the answer given the question from GSM8K dataset.
    """
    # Load GSM8K dataset from HuggingFace
    ds = load_dataset("gsm8k", "main")
    # Convert to pandas DataFrame
    df = ds["train"].to_pandas()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 1 to the seed to avoid the same seed for question and answer
    args["seed"] += 1

    # Generate answers given questions
    generator = GenericMultivariate(
        df=df,
        name="gsm8k_answer_from_question",
        given_variables=["question"],
        gen_variables=["answer"],
        descriptions=[
            "Given a math word problem, generate its step-by-step solution with calculations and reasoning.",
            "Solve grade school math word problems by providing detailed worked solutions.",
            "Given: question (math word problem). Generate: answer (step-by-step solution with final numerical answer).",
        ],
    )

    return generator.generate_many(**args)


def generate_gsm8k_question_answer(**kwargs) -> pd.DataFrame:
    """
    Generate both question and answer jointly from GSM8K dataset.
    """
    # Load GSM8K dataset from HuggingFace
    ds = load_dataset("gsm8k", "main")
    # Convert to pandas DataFrame
    df = ds["train"].to_pandas()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 2 to the seed to avoid the same seed for question and answer
    args["seed"] += 2

    # Generate both questions and answers
    generator = GenericMultivariate(
        df=df,
        name="gsm8k_question_answer",
        given_variables=[],
        gen_variables=["question", "answer"],
        descriptions=[
            "Generate both a math word problem (question) and its step-by-step solution (answer). These are grade school math problems covering arithmetic, basic algebra, and word problem reasoning.",
            "Create complete math problem pairs: generate both the word problem statement and its detailed solution with calculations.",
            "Generate: question (math word problem), answer (step-by-step solution with final numerical answer).",
        ],
    )

    return generator.generate_many(**args)


def generate_gsm8k_question_from_answer(**kwargs) -> pd.DataFrame:
    """
    Generate the question given the answer from GSM8K dataset.
    """
    # Load GSM8K dataset from HuggingFace
    ds = load_dataset("gsm8k", "main")
    # Convert to pandas DataFrame
    df = ds["train"].to_pandas()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 3 to the seed to avoid the same seed for other generation tasks
    args["seed"] += 3

    # Generate questions given answers
    generator = GenericMultivariate(
        df=df,
        name="gsm8k_question_from_answer",
        given_variables=["answer"],
        gen_variables=["question"],
        descriptions=[
            "Given a step-by-step math solution, generate a word problem that would lead to this solution.",
            "Create math word problems that match the given worked solution and final answer.",
            "Given: answer (step-by-step solution). Generate: question (word problem that leads to this solution).",
        ],
    )

    return generator.generate_many(**args)
