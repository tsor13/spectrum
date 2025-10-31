import ast
import os
import re
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def globaloqa_loader() -> pd.DataFrame:
    """
    Load the GlobalOQA data.
    """
    df = pd.read_csv(
        "hf://datasets/Anthropic/llm_global_opinions/data/global_opinions.csv"
    )

    # parse selections string into a Python dict
    def _parse_defaultdict(s: str) -> dict:
        m = re.search(r"\{.*\}", s)
        return ast.literal_eval(m.group()) if m else {}

    df["selections"] = df["selections"].apply(_parse_defaultdict)

    # convert selections dict to a list of dicts
    df["selections_list"] = df["selections"].apply(
        lambda d: [{"country": k, "probabilities": v} for k, v in d.items()]
    )
    # question      When it comes to Germany’s decision-making in ...
    # selections    defaultdict(<class 'list'>, {'Belgium': [0.21,...
    # options       ['Has too much influence', 'Has too little inf...
    # source                                                      GAS
    df["options"] = df["options"].apply(ast.literal_eval)

    rows = []
    for _, row in df.iterrows():
        # loop throug
        # "defaultdict(<class 'list'>, {'Belgium': [0.21, 0.07, 0.69, 0.03], 'France': [0.35, 0.09, 0.54, 0.02], 'Germany': [0.13131313131313133, 0.30303030303030304, 0.5252525252525253, 0.04040404040404041], 'Greece': [0.86, 0.04, 0.1, 0.0], 'Italy': [0.6138613861386139, 0.0297029702970297, 0.3465346534653465, 0.009900990099009901], 'Netherlands': [0.2, 0.06, 0.72, 0.02], 'Spain': [0.53, 0.03, 0.43, 0.01], 'Sweden': [0.15, 0.02, 0.82, 0.01]})"
        for country_probs in row["selections_list"]:
            country = country_probs["country"]
            probabilities = country_probs["probabilities"]
            prob_dict = []
            for option, prob in zip(row["options"], probabilities):
                # prob_dict.append({option: prob})
                prob_dict.append({option: int(np.round(prob, 2) * 100)})
            rows.append(
                {
                    "question": row["question"],
                    "country": country,
                    "options": row["options"],
                    "distribution": prob_dict,
                    "source": row["source"],
                }
            )
    df = pd.DataFrame(rows)
    return df


def generate_globaloqa_data(**kwargs) -> pd.DataFrame:
    """
    Generate the GlobalOQA data.
    """
    args = {
        "seed": 42,
        "n_per": 2,
    }
    args.update(kwargs)
    df = globaloqa_loader()
    # convert question, options, distribution to strings
    df["question"] = df["question"].astype(str)
    df["options"] = df["options"].astype(str)
    df["distribution"] = df["distribution"].astype(str)
    dfs = []
    for country in df["country"].unique():
        country_df = df[df["country"] == country]
        # do generic multivariate
        generic_multivariate = GenericMultivariate(
            country_df,
            given_variables=["question", "options"],
            gen_variables=["distribution"],
            name=f"globaloqa_{country}",
            descriptions=[
                "People from an unknown location were surveyed. For each question, predict the percentage of people who chose each option. (list of dicts)",
                "Country: {country}\nFor each question, predict the percentage of people from the country who chose each option. (list of dicts)",
            ],
        )
        # bump up seed by 1
        args["seed"] += 1
        dfs.append(generic_multivariate.generate_many(**args))
    return pd.concat(dfs)
