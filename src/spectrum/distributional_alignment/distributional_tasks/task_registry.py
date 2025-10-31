import os
import sys

from spectrum.distributional_alignment.distributional_tasks.global_oqa import (
    load_globaloqa_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.habermas import (
    load_habermas_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.mpi import load_mpi_dataset
from spectrum.distributional_alignment.distributional_tasks.numbergame import (
    load_numbergame_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.nytimes import (
    load_nytimes_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.rotten_tomatoes import (
    load_rotten_tomatoes_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.states import (
    load_states_dataset,
)
from spectrum.distributional_alignment.distributional_tasks.urn import load_urn_dataset

default_data_map = {
    "global_oqa": load_globaloqa_dataset,
    "nytimes": load_nytimes_dataset,
    "mpi": load_mpi_dataset,
    "rotten_tomatoes": load_rotten_tomatoes_dataset,
    "urn": load_urn_dataset,
    "habermas": load_habermas_dataset,
    "numbergame": load_numbergame_dataset,
    "states": load_states_dataset,
}
