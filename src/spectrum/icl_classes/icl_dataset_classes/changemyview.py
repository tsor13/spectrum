import json
import os
import random
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID

SEED = 42


def sample_jsonl_reservoir(
    file_path: str, sample_size: int, seed: int = SEED
) -> pd.DataFrame:
    """
    Sample exactly `sample_size` lines from a JSONL file at random
    using reservoir sampling (single‐pass, memory O(sample_size)).
    """
    random.seed(seed)
    reservoir = []  # will hold our sampled records

    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            record = json.loads(line)
            if idx <= sample_size:
                # Fill up the reservoir array
                reservoir.append(record)
            else:
                # Replace elements with gradually decreasing probability
                j = random.randint(1, idx)
                if j <= sample_size:
                    reservoir[j - 1] = record

    # Convert list of dicts to a DataFrame
    return pd.DataFrame(reservoir)


def load_data():

    # check cache on the function object
    if getattr(load_data, "_data", None):
        # we have a hit!
        print(">>> Re-using cached DataFrame")
        return load_data._data

    df = sample_jsonl_reservoir("data/changemyview/threads.jsonl", sample_size=10000)

    # remove any “> *...This is a footnote…” line
    # (?m) = multi-line mode: ^/$ match per line
    footnote_re = r"""(?m)(?i)^(?:>|\&gt;)\s*\*.*?This is a footnote.*?$\n?"""

    df["selftext"] = df["selftext"].str.replace(footnote_re, "", regex=True)
    ignore_strings = set(["", "[deleted]", "[removed]"])

    df = df[
        df["selftext"].notnull()
        & ~df["selftext"].isin(ignore_strings)
        & df["title"].notnull()
        & ~df["title"].isin(ignore_strings)
        & (df["num_comments"] > 4)
    ]

    df["post"] = df["title"] + "\n\n" + df["selftext"]

    def filter_comments(comments):
        as_list = [
            c
            for c in comments
            if c.get("body")
            and c.get("score") is not None
            and c.get("body") not in ignore_strings
        ]
        if not as_list:
            return None
        result = pd.DataFrame(as_list)
        return result

    df["comments"] = df["comments"].apply(filter_comments)

    df = df[df["comments"].notnull()]

    # change score to string
    df["score_str"] = df["score"].astype(str)

    # store in cache on the function
    load_data.data = df

    # strip whitespace from the used string columns
    for col in ["post", "title", "selftext"]:
        df[col] = df[col].str.strip()

    return df


def generate_cmv_categories(**kwargs) -> pd.DataFrame:
    """
    Generate prompt and title categories from the cmv data.
    """
    df = load_data()

    args = {
        "seed": SEED,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=16),
    }
    args.update(kwargs)

    dfs = []

    #########################
    dist = SingleVariableIID(
        categories=df["post"].unique(),
        name=f"cmv_questions",
        replacement=False,
        descriptions=["Posts about which one might want to change their view"],
    )
    dfs.append(dist.generate_many(**args))

    #########################

    dist = SingleVariableIID(
        categories=df["title"].unique(),
        name=f"cmv_title",
        replacement=False,
        descriptions=["Post titles about which one might want to change their view"],
    )
    dfs.append(dist.generate_many(**args))

    #########################

    data = pd.concat(dfs)
    return data


def generate_cmv_posts(**kwargs) -> pd.DataFrame:
    """
    Generate prompt and title categories from the cmv data.
    """
    df = load_data()

    args = {
        "seed": 42,
        "n_per": -1,
        "max_total": 500,
        "n_iter": geom_and_poisson_iter(
            mean=16
        ),  # TAYLOR - many fewer fit into context window
    }
    args.update(kwargs)
    # add one to the seed
    args["seed"] += 1

    dfs = []

    #########################
    dist = GenericMultivariate(
        df=df,
        name=f"cmv_post_from_title",
        given_variables=["title"],
        gen_variables=["selftext"],
        descriptions=["The post that matches the title"],
    )
    dfs.append(dist.generate_many(**args))

    #########################
    dist = GenericMultivariate(
        df=df,
        name=f"cmv_title_from_post",
        given_variables=["selftext"],
        gen_variables=["title"],
        descriptions=["The title that matches the post"],
    )
    dfs.append(dist.generate_many(**args))
    #########################

    dist = GenericMultivariate(
        df=df,
        name=f"cmv_score_from_post",
        given_variables=["title", "selftext"],
        gen_variables=["score"],
        descriptions=["The score (upvotes - downvotes) on the post"],
    )
    dfs.append(dist.generate_many(**args))
    #########################

    data = pd.concat(dfs)
    return data


def generate_cmv_comments(**kwargs) -> pd.DataFrame:
    """
    Generate comments and related items from the cmv data.
    """
    # TAYLOR - deprecating this for now as it is quite long context and hard to fit into the model during training
    raise NotImplementedError("Comments are deprecated for now")
