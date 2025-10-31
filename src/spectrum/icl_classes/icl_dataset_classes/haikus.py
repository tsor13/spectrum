import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID

"""
To download data:
```
cd data
git clone git@github.com:docmarionum1/haikurnn.git
cd -
```
"""


def load_haikus_helper(folder: str | None = None):
    if folder is None:
        folder = "data/haikurnn/input/poems/"
    # read haikus.csv
    df = pd.read_csv(f"{folder}/haikus.csv")
    # make poem column which is 0 + 1 + 2 with newlines inbetween
    df["poem"] = df.apply(lambda row: f"{row['0']}\n{row['1']}\n{row['2']}", axis=1)

    # clean_syllable
    def clean_syllable_str(syllable_str):
        # add [ and ] to the beginning and end
        s = syllable_str.strip()
        s = "[ " + s + " ]"
        # eval the string
        s = eval(s)
        # keep first element
        return str(s[0])

    df["0_clean"] = df["0_syllables"].apply(clean_syllable_str)
    df["1_clean"] = df["1_syllables"].apply(clean_syllable_str)
    df["2_clean"] = df["2_syllables"].apply(clean_syllable_str)
    df["syllables"] = df.apply(
        lambda row: ", ".join([row["0_clean"], row["1_clean"], row["2_clean"]]), axis=1
    )
    return df


def generate_haikus(folder: str | None = None, **kwargs):
    args = {
        "seed": 42,
        # "n_per": 10,
        # "n_iter": 100,
        "n_per": -1,
        "n_iter": 64,
        "max_total": 200,
    }
    args.update(kwargs)

    df = load_haikus_helper(folder)
    df = df[["source", "poem", "syllables"]]
    # do individual multivariate where source is the ind
    dfs = []

    # generate poem from data source using syllables as given
    dfs.append(
        IndividualMultivariate(
            df,
            individual_id_column="source",
            given_variables=["syllables"],
            gen_variables=["poem"],
            name="haikus-syllables",
            descriptions=["haikus.\npoem ~ syllables"],
        ).generate_many(**args)
    )
    # increment seed
    args["seed"] += 1
    # now do targeted generation
    df["include_word"] = df.poem.apply(lambda x: np.random.choice(x.split()))
    # generate poem from data source using include_word as given
    dfs.append(
        IndividualMultivariate(
            df,
            individual_id_column="source",
            given_variables=["include_word", "syllables"],
            gen_variables=["poem"],
            name="haikus-includeword",
            descriptions=["haikus.\npoem ~ include_word, syllables"],
        ).generate_many(**args)
    )
    # increment seed
    args["seed"] += 1

    # do straight up poem generation no given varibales
    dfs.append(
        SingleVariableIID(
            df["poem"].unique().tolist(),
            descriptions=["haikus."],
            name="haikus-egn",
            replacement=False,
        ).generate_many(**args)
    )
    return pd.concat(dfs)
