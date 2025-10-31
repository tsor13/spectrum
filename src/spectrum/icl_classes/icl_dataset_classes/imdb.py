import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_imdb(inverse=False, **kwargs) -> pd.DataFrame:
    imdb = pd.read_parquet(
        "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet",
        engine="pyarrow",
    )

    args = {
        "seed": 42,
        "n_per": 200,
        "n_iter": geom_and_poisson_iter(mean=128),
    }
    args.update(kwargs)
    if inverse:
        imdb["rating"] = imdb["label"].apply(
            lambda x: "positive" if x == 0 else "negative"
        )
        imdb = imdb.rename(columns={"text": "review"})
    else:
        imdb["rating"] = imdb["label"].apply(
            lambda x: "negative" if x == 0 else "positive"
        )
        imdb = imdb.rename(columns={"text": "review"})

    dfs = []
    ########

    generic_multivariate = GenericMultivariate(
        df=imdb,
        given_variables=["review"],
        gen_variables=["rating"],
        name="imdb_inverse",
        descriptions=[
            "Classify whether the following movie reviews as negative or positive."
        ],
    )
    dfs.append(generic_multivariate.generate_many(**args))

    ########

    return pd.concat(dfs)


def generate_imdb_individual(p_inverse=0.05, **kwargs) -> pd.DataFrame:
    """
    Generate the prompts from the prism data.
    """
    raise DeprecationWarning("This function is deprecated. Use generate_imdb instead.")
    imdb = pd.read_parquet(
        "hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet",
        engine="pyarrow",
    )
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=20),
        "max_inds": 1000,
        # "max_inds": 400,
    }
    args.update(kwargs)
    # shuffle imdb
    imdb = imdb.sample(frac=1, random_state=args["seed"]).reset_index(drop=True)
    # add annotator_id column, which groups every n_per together
    imdb["annotator_id"] = imdb.index // args["n_per"]
    # split into positive and negative
    # randomly determine if each annotator_id is inverse
    annotator_inverse = np.random.binomial(
        1, p_inverse, len(imdb["annotator_id"].unique())
    )
    # add annotator_inverse column
    imdb["annotator_inverse"] = imdb["annotator_id"].apply(
        lambda x: annotator_inverse[x]
    )

    # apply label to each row:
    def apply_label(row):
        if row["annotator_inverse"]:
            return "positive" if row["label"] == 0 else "negative"
        else:
            return "negative" if row["label"] == 0 else "positive"

    imdb["label"] = imdb.apply(apply_label, axis=1)
    # rename text to review
    imdb = imdb.rename(columns={"text": "review"})
    # Individual Multivariate w/ annotator_id as individual_id_column
    generic_multivariate = IndividualMultivariate(
        df=imdb,
        individual_id_column="annotator_id",
        given_variables=["review"],
        gen_variables=["label"],
        name="imdb_individual",
    )
    data = generic_multivariate.generate_many(**args)
    return data
