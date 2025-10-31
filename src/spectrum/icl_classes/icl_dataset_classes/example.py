"""
An scaffold and some basic explainability of how to contribute a new dataset!
"""
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

# add any instructions to download the data here if necessary
"""
To download data:
```
mkdir data/baby_names
cd data/baby_names
...
cd -
```
"""

# can have as many helper functions as you want here

def generate_****( # you will import this into dataloaders.py
    # any dataset specific args first
    **kwargs # for overriding generate_many defaults
):
    args = { # decent generate many defaults for getting somewhere between 10-1000 training data examples (rows), such that we don't calculate loss on the same data more than once or maybe twice
        "seed": 42, # required for every dataset for reproducibility. Can also use in this function if you like.
        # "n_per": 1, # number of generations to do per example. Normally this should be 1, but if you have a giant dataset that won't fit into context, you could sample more times.
        # "n_iter": None, # if there are many draws, the maximum number of draws to include. Can be an int, or an iter that randomly samples numbers (see below)
        # # "n_iter": geom_and_poisson_iter(128), # for each sample, will randomly sample a number of examples to include with a mean of 128
        # "max_inds": 5000, the maximum number of examples to include from the dataset. If you need to create multiple random datasets (e.g., one for each user), can also use that in this function to mean maximum number of users to draw.
        # there are other variables, but no need to touch them here! :)
    }
    args.update(kwargs)

    # load or generate data

    # Load or generate your data here

    # Choose the appropriate generator class based on your data type:
    # - SingleVariableIID: for categorical data generation
    # - GenericMultivariate: for predicting variables from other variables
    # - IndividualMultivariate: for predicting variables grouped by individual
    return generator.generate_many(**args)