"""
uv run random_classes/dataset_loaders/mmlu.py
"""

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_mmlu(inverse=False, **kwargs) -> pd.DataFrame:
    # cais/mmlu
    df = load_dataset("cais/mmlu", "all")["test"].to_pandas()

    seed = kwargs.get("seed", 42)
    np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)

    letters = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

    # make template of Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {answer}
    def template(row):
        text = f"Question: {row['question']}"
        for i, choice in enumerate(row["choices"]):
            text += f"\n{letters[i]}. {choice}"
        return text

    df["prompt"] = df.apply(template, axis=1)
    df["answer_letter"] = df["answer"].apply(lambda x: letters[x])

    args = {
        "seed": 42,
        "n_per": 1000,
        "n_iter": 20,
    }
    args.update(kwargs)

    dataset = GenericMultivariate(
        df=df,
        given_variables=["prompt"],
        gen_variables=["answer_letter"],
        name="mmlu",
        descriptions=[
            "Given a question and a list of choices, return the correct answer choice."
        ],
    )
    data = dataset.generate_many(**args)

    return data
