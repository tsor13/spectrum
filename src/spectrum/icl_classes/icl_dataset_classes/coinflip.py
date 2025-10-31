import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.bernoulli import Bernoulli
from spectrum.icl_classes.icl_class import geom_and_poisson_iter


def generate_coinflip_data(n_ps: int = 1000, **kwargs) -> pd.DataFrame:
    args = {
        "seed": 42,
        "n_iter": geom_and_poisson_iter(mean=128),
    }
    args.update(kwargs)
    np.random.seed(args["seed"])
    # generate n_ps ps
    ps = np.random.uniform(0, 1, n_ps)

    all_data = []
    for i, p in enumerate(ps):
        bernoulli = Bernoulli(
            p=p, random_seed=i + args["seed"], sample_names=["Heads", "Tails"]
        )

        args["seed"] += 1
        data = bernoulli.generate_many(**args)
        all_data.append(data)
    data = pd.concat(all_data)
    return data
