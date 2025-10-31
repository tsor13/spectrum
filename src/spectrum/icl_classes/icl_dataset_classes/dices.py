"""
uv run src/icl_classes/icl_dataset_classes/dices.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

"""
To download data:
```
cd data
git clone git@github.com:google-research-datasets/dices-dataset.git
cd -
```
"""


def load_dices_990(folder: str | None = None):
    if folder is None:
        folder = "data/dices-dataset"
    data = pd.read_csv(
        os.path.join(folder, "990/diverse_safety_adversarial_dialog_990.csv")
    )
    demographic_cols = [
        "rater_gender",
        "rater_locale",
        "rater_race",
        "rater_race_raw",
        "rater_age",
        "rater_education",
    ]
    question_cols = ["Q_overall"]
    return {
        "demographic_cols": demographic_cols,
        "question_cols": question_cols,
        "data": data,
    }


def load_dices_350(folder: str | None = None):
    if folder is None:
        folder = "data/dices-dataset"
    demographic_cols = [
        "rater_gender",
        "rater_race",
        "rater_raw_race",
        "rater_age",
        "rater_education",
    ]
    question_cols = ["Q_overall"]
    data = pd.read_csv(
        os.path.join(folder, "350/diverse_safety_adversarial_dialog_350.csv")
    )
    return {
        "demographic_cols": demographic_cols,
        "question_cols": question_cols,
        "data": data,
    }


def generate_dices(**kwargs):
    args = {
        "seed": 42,
        "n_iter": 100,
        "max_inds": 500,
    }
    args.update(kwargs)
    dfs = []
    for dataset in ["990", "350"]:
        if "990" in dataset:
            out = load_dices_990()
        else:
            out = load_dices_350()
        data = out["data"]
        demographic_cols = out["demographic_cols"]
        question_cols = out["question_cols"]
        # get unique rater_ids
        rater_ids = data["rater_id"].unique()
        # shuffle rater_ids
        np.random.seed(args["seed"])
        np.random.shuffle(rater_ids)
        # take first n_users
        rater_ids = rater_ids[: args["max_inds"]]
        for rater_id in rater_ids:
            rater_data = data[data["rater_id"] == rater_id]
            # get demographics
            rater_demographics = rater_data.iloc[0][demographic_cols]
            if len(rater_data.dropna(subset=demographic_cols)):
                rater_demographics = rater_data.iloc[0][demographic_cols]
            # to string
            rater_demographics = str(rater_demographics.to_dict())
            # do generic multivariate
            dfs.append(
                GenericMultivariate(
                    df=rater_data,
                    given_variables=["context", "response"],
                    gen_variables=question_cols,
                    name=f"dices_individual",
                    descriptions=[
                        "Given a context conversation and a response from a language model, the rater was tasked to label whether or not they thought the last dialog model response is unsafe. (Q_overall: Yes, No, Unsure)",
                        "Given a context conversation and a response from a language model, the rater was tasked to label whether or not they thought the last dialog model response is unsafe. (Q_overall: Yes, No, Unsure)\nRater demographics: "
                        + rater_demographics,
                    ],
                ).generate_many(**args)
            )
    return pd.concat(dfs)


if __name__ == "__main__":
    data = generate_dices()
