"""
uv run random_classes/dataset_loaders/chemistry.py
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate

# [tsor13@g3085 bayesbench]$ ls data/chemistry/
# esol_iupac.csv  oxidative_methane_coupling.csv  train_data_num_feats.csv


def read_esol_iupac():
    df = pd.read_csv("data/chemistry/esol_iupac.csv")
    return df


def read_oxidative_methane_coupling():
    df = pd.read_csv("data/chemistry/train_data_num_feats.csv")
    return df


def generate_chemistry_esol(
    # config: str = "default",
    # split: str = "validation",
    # shuffle_dataset: bool = True,
    # inverse: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    df = read_esol_iupac()
    # shuffle with seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Compound ID                                                  1,1,1,2-Tetrachloroethane
    # measured log(solubility:mol/L)                                                   -2.18
    # ESOL predicted log(solubility:mol/L)                                            -2.794
    # SMILES                                                                  ClCC(Cl)(Cl)Cl
    # SELFIES                                 [Cl][C][C][Branch1][C][Cl][Branch1][C][Cl][Cl]
    # InChI                                               InChI=1S/C2H2Cl4/c3-1-2(4,5)6/h1H2
    # IUPAC                                                        1,1,1,2-tetrachloroethane

    # do multivariate to predict measured log(solubility:mol/L) from SMILES, SELFIES, InChI, IUPAC

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 15,
        "permutations_per_iter": 5,  # since we're using for eval, want to have some variety
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=df,
        given_variables=["SMILES", "SELFIES", "InChI", "IUPAC"],
        gen_variables=["measured log(solubility:mol/L)"],
        # name=f"drop_{config}_{split}",
        name=f"chemistry_esol",
        descriptions=[
            "Predict the measured log(solubility:mol/L) from SMILES, SELFIES, InChI, IUPAC",
        ],
    )
    # "Provide only the answer extracted from the passage."])

    return dataset_wrapper.generate_many(**args)


def generate_chemistry_oxidative(seed: int = 42, **kwargs: Any) -> pd.DataFrame:
    df = read_oxidative_methane_coupling()
    # shuffle with seed
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    args = {
        "seed": seed,
        "n_per": -1,
        "n_iter": 10,
    }
    args.update(kwargs)

    dataset_wrapper = GenericMultivariate(
        df=df,
        given_variables=["prompt"],
        gen_variables=["C2_yield"],
        name=f"chemistry_oxidative",
        descriptions=[
            "The following is data from a set of chemistry experiments. Predict the C2_yield from the experiment description."
        ],
    )
    return dataset_wrapper.generate_many(**args)
