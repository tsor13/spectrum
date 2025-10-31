import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def valueprism_loader() -> pd.DataFrame:
    """
    Load the ValuePrism data.
    """
    df = pd.read_csv("hf://datasets/allenai/ValuePrism/full/full.csv")
    return df


def generate_valueprism_situations(**kwargs) -> pd.DataFrame:
    """
    Generate the situations from the ValuePrism data.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=256),
        "max_inds": 1,
    }
    args.update(kwargs)
    df = valueprism_loader()
    # get unique situations
    situations = df["situation"].unique().tolist()
    # make categorical
    categorical = SingleVariableIID(
        situations,
        name="valueprism_situation",
        descriptions=["Generate a situation.", "Situations to do moral reasoning over"],
        replacement=False,
    )
    data = categorical.generate_many(**args)
    return data


def generate_valueprism_vrd_data(**kwargs) -> pd.DataFrame:
    """
    Generate the VRD data from the ValuePrism data.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 1,
        "max_total": 500,
    }
    args.update(kwargs)
    df = valueprism_loader()

    def row_to_vrd(row: pd.Series) -> Dict[str, Any]:
        """
        Convert a row of the ValuePrism data to a VRD.
        """
        vrd = row["vrd"]
        text = row["text"]
        explanation = row["explanation"]
        valence = row["valence"]
        # return f"{text}({vrd}): {explanation} ({valence})"
        d = {
            "text": text,
            "vrd": vrd,
            "explanation": explanation,
            "valence": valence,
        }
        return d

    df["vrd_text"] = df.apply(row_to_vrd, axis=1)
    # shuffle
    df = df.sample(frac=1, random_state=args["seed"]).reset_index(drop=True)
    # aggregate by situation and concat vrd_text
    df = df.groupby("situation")["vrd_text"].apply(lambda x: str(list(x))).reset_index()
    # rename vrd_text to output
    df.rename(columns={"vrd_text": "output"}, inplace=True)
    # make generic multivariate
    description_text = "You will be given a situation and be asked to do moral reasoning over it. For each situation, you will output a list of dictionaries, each containing the following keys:\n-text: A relevant moral consideration or factor.\n-vrd: Whether the consideration is a value, a right, or a duty.\n-explanation: A short explanation of why the consideration is relevant to the situation.\n-valence: The valence of the consideration (Supports, Opposes, or Neutral). It should ideally contain at least 3-8 moral considerations."
    generic_multivariate = GenericMultivariate(
        df=df,
        given_variables=["situation"],
        gen_variables=["output"],
        name="valueprism_vrd",
        descriptions=[description_text],
    )
    data = generic_multivariate.generate_many(**args)
    return data


def generate_valueprism_vrds_noncontextual(**kwargs) -> pd.DataFrame:
    """
    Generate the VRDs from the ValuePrism data.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=256),
    }
    args.update(kwargs)
    df = valueprism_loader()
    dfs = []
    # get unique values
    np.random.seed(args["seed"])
    for vrd in ["Value", "Right", "Duty"]:
        values = df[df["vrd"] == vrd]["text"].unique().tolist()
        # convert all to string
        values = [str(v) for v in values]
        # shuffle values
        np.random.shuffle(values)
        dfs.append(
            SingleVariableIID(
                values,
                name=f"valueprism_vrds_{vrd}",
                descriptions=[f"{vrd} (wrt moral reasoning)"],
                replacement=False,
            ).generate_many(**args)
        )
    data = pd.concat(dfs)
    return data


def generate_valueprism_misc(
    max_values: int = 400, min_val_occurrences: int = 10, **kwargs
) -> pd.DataFrame:
    """
    Generate the situations from the ValuePrism data.
    """
    args = {
        "seed": 42,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 1,
    }
    args.update(kwargs)
    df = valueprism_loader()
    dfs = []
    # get unique situations
    # values = df[df['vrd'] == 'Value']['text'].unique().tolist()
    value_counts = df[df["vrd"] == "Value"]["text"].value_counts()
    values = value_counts[value_counts >= min_val_occurrences].index.tolist()
    # shuffle values
    np.random.seed(args["seed"])
    np.random.shuffle(values)

    for i, value in enumerate(values[:max_values]):
        df_value = df[df["text"] == value]
        # get unique situations
        situations = df_value["situation"].unique().tolist()
        # make categorical
        dfs.append(
            SingleVariableIID(
                situations,
                name=f"valueprism_misc_{value}",
                descriptions=[f"Generate a situation relating to {value}."],
                replacement=False,
            ).generate_many(**args)
        )

    data = pd.concat(dfs)
    return data
