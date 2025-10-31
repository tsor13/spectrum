import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from spectrum.icl_classes.single_variable_iid import SingleVariableIID

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("Paul/IssueBench", "issues")


def load_issuebench():
    ds = load_dataset("Paul/IssueBench", "prompts")
    # convert to pandas
    # df = ds.to_pandas()
    # pandas_dict = {split: ds[split].to_pandas() for split in ds}
    df = ds["prompts_full"].to_pandas()
    topics = df["topic_text"].unique()
    templates = df["template_text"].unique()
    return {
        "topics": topics,
        "templates": templates,
    }


def generate_issuebench_topics(**kwargs):
    args = {
        "seed": 42,
        "n_per": 2,
    }
    args.update(kwargs)
    topics = load_issuebench()["topics"]
    # make categorical
    topics = SingleVariableIID(
        topics,
        name="issuebench_topics",
        descriptions=[f"Topics relating to politics"],
        replacement=False,
    )
    return topics.generate_many(**args)


def generate_issuebench_templates(**kwargs):
    args = {
        "seed": 42,
        "n_per": 2,
    }
    args.update(kwargs)
    templates = load_issuebench()["templates"]
    templates = SingleVariableIID(
        templates,
        name="issuebench_templates",
        descriptions=[
            f"Templates to use for a language model benchmark about bias on a topic. The topic should be in the template as X."
        ],
        replacement=False,
    )
    return templates.generate_many(**args)


def generate_issuebench(**kwargs):
    topics = generate_issuebench_topics(**kwargs)
    templates = generate_issuebench_templates(**kwargs)
    return pd.concat([topics, templates])
