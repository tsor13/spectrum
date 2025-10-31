import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.single_variable_iid import SingleVariableIID

"""
To download data:
```
mkdir data/baby_names
cd data/baby_names
wget https://www.ssa.gov/oact/babynames/names.zip
unzip names.zip
rm names.zip
cd -
```
"""
# Cache to store loaded dataframes
_baby_names_cache = {}


def load_baby_names_helper(year):
    if year not in _baby_names_cache:
        df = pd.read_csv(
            f"data/baby_names/yob{year}.txt",
            header=0,
            names=["name", "sex", "frequency"],
        )
        _baby_names_cache[year] = df
    return _baby_names_cache[year]


def generate_babynames_helper(
    top_k=10, timespan=2021, sex="balanced", sampling_strategy="uniform"
):
    """
    :timespan: should either be:
        - "all": indicating we should take all years in the data
        - int: indicating that we should take a single year
        - list: containing the list of years we care about
        - range: containing the range of years we care about
    :sex: should either be "M", "F", "balanced" (in which top_k must be even), or "unbalanced" (where top_k is taken indpendent of sex. Boy's names are less diverse, so they usually dominate)
    """
    assert sex in {"unbalanced", "balanced", "M", "F"}
    assert sampling_strategy in {"uniform", "population"}
    year_range = None
    year_str = ""
    if timespan == "all":
        year_range = range(1880, 2024)
        year_str = "from 1880 - 2023"
    elif isinstance(timespan, int):
        year_range = range(timespan, timespan + 1)
        year_str = f"in {timespan}"
    else:
        year_range = timespan
        if isinstance(timespan, type(range(0, 1))):
            assert (
                timespan.step == 1
            ), "description assumes that the step of the range is 1"
            year_str = f"from {timespan.start} - {timespan.stop - 1}"
        else:
            year_str = ", ".join([str(yr) for yr in timespan])

    all_names = []
    for year in year_range:
        names = load_baby_names_helper(year)
        if sex not in {"unbalanced", "balanced"}:
            names = names[names["sex"] == sex]
        all_names.append(names)

    all_names_df = pd.concat(all_names)
    all_names_grouped = all_names_df.groupby(["name", "sex"]).sum()["frequency"]

    # following the SS data, we count the same name with different sex as different names
    if sex == "balanced":
        assert (
            top_k % 2 == 0
        ), "For balanced sexes we need the same number of boys and girls, so top_k must be even."
        top_names_male = (
            all_names_grouped[:, "M"]
            .sort_values(ascending=False)[: top_k // 2]
            .reset_index()
        )
        top_names_female = (
            all_names_grouped[:, "F"]
            .sort_values(ascending=False)[: top_k // 2]
            .reset_index()
        )
        top_names_df = pd.concat([top_names_male, top_names_female])
    else:
        top_names_df = all_names_grouped.sort_values(ascending=False)[
            :top_k
        ].reset_index()

    category_probs = None
    if sampling_strategy == "population":
        if sex == "balanced":
            # balance probability so P(M) == P(F) ==  50%, but conditional probabilities are accurate
            category_probs = (
                0.5 * top_names_male["frequency"] / top_names_male["frequency"].sum()
            ).tolist()
            category_probs.extend(
                (
                    0.5
                    * top_names_female["frequency"]
                    / top_names_female["frequency"].sum()
                ).tolist()
            )
        else:
            category_probs = top_names_df["frequency"] / top_names_df["frequency"].sum()

    top_names = top_names_df["name"].tolist()

    # US Social security data for gender is binary
    sex_str = ""  # for "balanced" and "unbalanced"
    if sex == "M":
        sex_str = "boy "  # note the trailing space!
    elif sex == "F":
        sex_str = "girl "  # note the trailing space!

    # description = f"The following are among the top {top_k} baby {sex_str}names in the US {year_str}"
    descriptions = []
    if sampling_strategy == "population":
        descriptions.append(
            f"The following are drawn randomly from among the top {top_k} baby {sex_str}names in the US {year_str}, proportional to their population frequency."
        )
    elif sampling_strategy == "uniform":
        # descriptions.append(f"The following is a list of the top {top_k} baby {sex_str}names in the US {year_str}")
        descriptions.append(
            f"The following are drawn from the list of top {top_k} baby {sex_str}names in the US {year_str}"
        )
    else:
        raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
    name = f"babynames_top_k={top_k}_ts={timespan}_sex={sex}_sampling_strat={sampling_strategy}"
    # print(name)
    return SingleVariableIID(
        top_names,
        category_probs=category_probs,
        name=name,
        descriptions=descriptions,
        # replacement=(sampling_strategy=="population"),
        replacement=False,
    )


def generate_babynames(
    sex_list=["balanced", "M", "F"],
    sampling_strategy_list=["uniform", "population"],
    top_k_list=[10, 100, 1000, 5000],
    # timespan_list = None, # ['all'] + list(range(1880, 2024)),
    timespan_list=["all"] + list(range(1880, 2024))[::-1],
    randomize_args=True,
    **kwargs,
):
    # copied from `diffuse_distributions`. No idea if this is right.
    args = {
        # "n_per": 1,
        "seed": 42,
        # "n_iter": None, # include all names
        "n_iter": 5000,  # maximum number of names to sample in replacement case
        "max_inds": 500,
        # "max_n": 5000, # maximum number of names to sample
    }
    args.update(kwargs)
    dfs = []

    def draw_zipfian(n):
        # sample p between 0 and 1
        p = np.random.rand()
        # make arange
        arange = np.arange(n) + 1
        inverse_rank = 1 / arange
        return (
            np.random.choice(arange, size=1, p=inverse_rank / inverse_rank.sum()).item()
            - 1
        )

    def draw_linear_reverse_rank(n):
        p = np.random.rand()
        arange = (np.arange(n) + 1)[::-1]
        probs = arange / arange.sum()
        return np.random.choice(np.arange(n), size=1, p=probs).item()

    # randomly choose between all combinations
    if randomize_args:
        np.random.seed(args["seed"])
        for i in tqdm(range(args["max_inds"])):
            np.random.seed(args["seed"] + i)
            sex = sex_list[np.random.randint(0, len(sex_list))]
            sampling_strategy = sampling_strategy_list[
                np.random.randint(0, len(sampling_strategy_list))
            ]
            top_k = top_k_list[np.random.randint(0, len(top_k_list))]
            # draw timespan from a zipfian
            # timespan = timespan_list[np.random.randint(0, len(timespan_list))]
            # timespan = timespan_list[draw_zipfian(len(timespan_list))]
            timespan = timespan_list[draw_linear_reverse_rank(len(timespan_list))]
            gen = generate_babynames_helper(
                sex=sex,
                sampling_strategy=sampling_strategy,
                top_k=top_k,
                timespan=timespan,
            )
            data = gen.generate_many(**args)
            # add hparams
            data["sex"] = sex
            data["sampling_strategy"] = sampling_strategy
            data["top_k"] = top_k
            data["timespan"] = str(timespan)
            dfs.append(data)
    else:  # do all combinations
        loop = tqdm(
            total=len(sex_list)
            * len(sampling_strategy_list)
            * len(top_k_list)
            * len(timespan_list)
        )
        for sex in sex_list:
            for sampling_strategy in sampling_strategy_list:
                for top_k in top_k_list:
                    for timespan in timespan_list:
                        gen = generate_babynames_helper(
                            sex=sex,
                            sampling_strategy=sampling_strategy,
                            top_k=top_k,
                            timespan=timespan,
                        )
                        data = gen.generate_many(**args)
                        # add hparams
                        data["sex"] = sex
                        data["sampling_strategy"] = sampling_strategy
                        data["top_k"] = top_k
                        data["timespan"] = str(timespan)
                        dfs.append(data)
                        loop.update(1)
    return pd.concat(dfs)
