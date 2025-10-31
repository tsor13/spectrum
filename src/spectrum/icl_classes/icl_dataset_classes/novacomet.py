import json
import os
import sys

import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_novacomet_dataset():
    dataset = load_dataset("tsor13/novacomet", split="train")
    # to pandas
    df = dataset.to_pandas()
    return df


def generate_novacomet_premise_data(
    file_name: str | None = None, **kwargs
) -> pd.DataFrame:
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # load from hugging face
    df = load_novacomet_dataset()
    # get premises
    premises = df["premise"].unique().tolist()
    # make categorical
    categorical = SingleVariableIID(
        premises,
        name="novacomet_hypothesis",
        descriptions=[
            "Generate an event.",
            "Events",
            "Generate a list of diverse events.",
            "Situations about which commonsense reasoning could be required.",
            "Single sentence events",
        ],
        replacement=False,
    )
    # sample n rows
    df = categorical.generate_many(**args)
    return df


def generate_novacomet_hypothesis_data(
    file_name: str | None = None, **kwargs
) -> pd.DataFrame:
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
    }
    args.update(kwargs)
    # add 1 to the seed to avoid the same seed for different tasks
    args["seed"] += 1
    df = load_novacomet_dataset()
    # groupby premise and put hypothesis in list
    df = df.groupby("premise")["hypothesis"].apply(list).reset_index()
    # make hypothesis a string
    # df["hypothesis"] = df["hypothesis"].apply(lambda x: "; ".join(x))
    df["hypothesis"] = df["hypothesis"].apply(json.dumps)
    # df["hypothesis"] = df["hypothesis"].apply(str)
    # rename to hypotheses
    df.rename(columns={"hypothesis": "hypotheses"}, inplace=True)
    # make generic multivariate
    generic_multivariate = GenericMultivariate(
        df=df,
        given_variables=["premise"],
        gen_variables=["hypotheses"],
        name="novacomet_premise",
        descriptions=[
            "Given a premise (or situation), generate a list of commonsense hypotheses separated by semicolons.",
            "You will be given a premise. Generate a list of around 8-10 hypotheses (or inferences) separated by semicolons.",
        ],
    )
    data = generic_multivariate.generate_many(**args)
    return data
