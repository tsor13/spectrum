import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_vc(**kwargs) -> pd.DataFrame:
    """
    Generate ______ from the value consistency data.
    """

    ds = load_dataset("jlcmoore/ValueConsistency")
    # convert huggingface dataset to pandas
    vc = ds["train"].to_pandas()

    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)

    vc_no_paraphrase = vc[~vc["rephrase"].astype(bool)]
    dfs = []
    dist = GenericMultivariate(
        df=vc_no_paraphrase,
        name=f"valueconsistency_questions",
        given_variables=["controversial", "language", "country"],
        gen_variables=["question"],
        descriptions=[
            "Value-based questions.",
            (
                "Value-based questions given whether "
                "the question is controversial, the language it is asked in, "
                "and the country it is for."
            ),
        ],
    )
    dfs.append(dist.generate_many(**args))

    data = pd.concat(dfs)
    return data
