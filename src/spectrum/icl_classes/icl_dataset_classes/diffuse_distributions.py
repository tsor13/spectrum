import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def get_diffuse_distributions(
    folder: str = "data/diffuse-distributions/prompts",
) -> List[Dict[str, Any]]:
    # recursive search for all json files
    dists = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    d = json.load(f)
                    # check if targets is in d
                    if "targets" in d:
                        # check if targets is a nonempty list
                        if len(d["targets"]) > 0:
                            dists[file] = d
    return dists


def categorical_from_diffuse_distribution(
    name: str, dist: Dict[str, Any]
) -> SingleVariableIID:
    request = dist["request"]
    # keep only the first sentence
    request = request.split(".")[0] + ", without replacement."
    targets = dist["targets"]
    remove_chars = ["{", "}", "\n", "\u200b"]
    for char in remove_chars:
        targets = [t.replace(char, "") for t in targets]
    targets = [t.strip() for t in targets]
    # remove empty strings
    targets = [t for t in targets if t]
    categorical = SingleVariableIID(
        targets, name=name, descriptions=[request], replacement=False
    )
    return categorical


def generate_diffuse_distributions_data(**kwargs) -> pd.DataFrame:
    args = {
        "seed": 42,
        "permutations_per_iter": 10,
    }
    args.update(kwargs)
    dists = get_diffuse_distributions()
    dfs = []
    for name, dist in dists.items():
        categorical = categorical_from_diffuse_distribution(name, dist)
        args["seed"] += 1
        dfs.append(categorical.generate_many(**args))
    data = pd.concat(dfs)
    return data


np.random.seed(42)
