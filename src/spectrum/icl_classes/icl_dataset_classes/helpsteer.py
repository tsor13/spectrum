import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_helpsteer_helper():
    splits = {"train": "train.jsonl.gz", "validation": "validation.jsonl.gz"}
    # load train
    df = pd.read_json("hf://datasets/nvidia/HelpSteer2/" + splits["train"], lines=True)
    # vector is helpfulness, correctness, coherence, complexity, verbosity
    df["pref_features"] = df[
        ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    ].values.tolist()
    rows = []
    # loop through unique prompts
    prompts = df["prompt"].unique()
    for prompt in prompts:
        # get all rows with this prompt
        prompt_rows = df[df["prompt"] == prompt]
        # get response1
        response1 = prompt_rows["response"].iloc[0]
        # get response2
        response2 = prompt_rows["response"].iloc[1]
        # get pref_features
        pref_features1 = prompt_rows["pref_features"].iloc[0]
        # get pref_features2
        pref_features2 = prompt_rows["pref_features"].iloc[1]
        # add to rows
        rows.append(
            {
                "prompt": prompt,
                "response1": response1,
                "response2": response2,
                "pref_features1": pref_features1,
                "pref_features2": pref_features2,
            }
        )
    # convert to dataframe
    df = pd.DataFrame(rows)
    # save to csv
    return df


def generate_helpsteer(
    # n_users: int = 200,
    **kwargs,  # for overriding generate_many defaults
):
    args = {  # decent generate many defaults for getting somewhere between 10-1000 training data examples (rows), such that we don't calculate loss on the same data more than once or maybe twice
        "seed": 42,  # required for every dataset for reproducibility. Can also use in this function if you like.
        "n_iter": 40,
        "max_inds": 500,
    }
    args.update(kwargs)

    # load or generate data
    df = load_helpsteer_helper()

    # set seed
    np.random.seed(args["seed"])
    # generate random user prefs (uniform from 0-1 for each feature)
    user_prefs = np.random.rand(args["max_inds"], 5)
    # shift the first 3 by 0.05, but the others by 0.5 (complexity and verbosity can be negative)
    user_prefs = user_prefs - np.array([0.05, 0.05, 0.05, 0.5, 0.5])
    dfs = []
    for i, user_pref in enumerate(user_prefs):
        # sample "n_iter" rows from df
        sampled_rows = df.sample(
            args["n_iter"], replace=False, random_state=args["seed"] + i
        )
        # add user_pref to sampled_rows
        # take dot with user_pref
        sampled_rows["user_pref1"] = sampled_rows["pref_features1"].apply(
            lambda x: np.dot(x, user_pref)
        )
        sampled_rows["user_pref2"] = sampled_rows["pref_features2"].apply(
            lambda x: np.dot(x, user_pref)
        )
        sampled_rows["winner"] = sampled_rows["user_pref1"] > sampled_rows["user_pref2"]
        # to 1 or 2
        sampled_rows["winner"] = sampled_rows["winner"].apply(lambda x: 1 if x else 2)
        # do generic multivariate
        user_pref_str = str(
            {
                k: float(np.round(v, 2))
                for k, v in zip(
                    [
                        "helpfulness",
                        "correctness",
                        "coherence",
                        "complexity",
                        "verbosity",
                    ],
                    user_pref,
                )
            }
        )
        dfs.append(
            GenericMultivariate(
                df=sampled_rows,
                given_variables=["prompt", "response1", "response2"],
                gen_variables=["winner"],
                name=f"helpsteer",
                descriptions=[
                    "The following are given: prompt, response1, response2. You must generate the winner.",
                    "The following are ratings from the same individual, who was asked to judge two language model outputs written by someone else. Generate the winner.",
                    "The following are ratings from the same individual, who was asked to judge two language model outputs written by someone else. Generate the winner.\nUser descriptionities: "
                    + user_pref_str,
                ],
            ).generate_many(**args)
        )
    # convert to dataframe
    df = pd.concat(dfs)
    return df
