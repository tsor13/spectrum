import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_community_alignment(sample_size=10_000):
    """Load the Facebook Community Alignment Dataset"""
    # Load streaming for efficiency during testing
    dataset = load_dataset(
        "facebook/community-alignment-dataset", split="filtered", streaming=True
    )

    # Take a sample for efficiency
    samples = []
    for i, sample in enumerate(dataset):
        if i >= sample_size:
            break
        if sample["first_turn_prompt"] and sample["first_turn_preferred_response"]:
            samples.append(sample)

    df = pd.DataFrame(samples)
    # replace <|end_of_text|> with "" in all responses
    df["first_turn_response_a"] = df["first_turn_response_a"].str.replace(
        "<|end_of_text|>", ""
    )
    df["first_turn_response_b"] = df["first_turn_response_b"].str.replace(
        "<|end_of_text|>", ""
    )
    df["first_turn_response_c"] = df["first_turn_response_c"].str.replace(
        "<|end_of_text|>", ""
    )
    df["first_turn_response_d"] = df["first_turn_response_d"].str.replace(
        "<|end_of_text|>", ""
    )

    return df


def generate_community_alignment_initial_prompt(**kwargs):
    """Generate data loader for all in-context prompts (multi-turn conversations)"""
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
    }
    args.update(kwargs)

    df = load_community_alignment()

    df = df.drop_duplicates(subset=["first_turn_prompt"])
    dfs = []
    for lang in df["assigned_lang"].unique():
        lang_df = df[df["assigned_lang"] == lang]
        prompts = lang_df["first_turn_prompt"].tolist()
        # strip
        prompts = [prompt.strip() for prompt in prompts]
        lang_data = SingleVariableIID(
            prompts,
            descriptions=[
                f"Generate a prompt for a language model in the following language: {lang}."
            ],
            name="community_alignment_initial_prompt",
            replacement=False,
        ).generate_many(**args)
        dfs.append(lang_data)
    data = pd.concat(dfs)
    return data


def row_to_user_demographics(row):
    d = {
        "age": row["annotator_age"],
        "gender": row["annotator_gender"],
        "education_level": row["annotator_education_level"],
        "annotator_political": row["annotator_political"],
        "ethnicity": row["annotator_ethnicity"],
        "country": row["annotator_country"],
    }
    # to json string
    return json.dumps(d)


def generate_community_alignment_individual_reply(**kwargs):
    """Generate data loader for first turn conversations only"""
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=25),
        "max_inds": 1000,
    }
    args.update(kwargs)

    df = load_community_alignment()

    # for each annotator, given first_turn_prompt and the first turn preferred response, output the second turn prompt
    df["model_response"] = df.apply(
        lambda row: row["first_turn_" + row["first_turn_preferred_response"]], axis=1
    )
    df["prompt"] = df.apply(lambda row: row["first_turn_prompt"], axis=1)
    df["reply"] = df["second_turn_prompt"]

    # drop so we have a minimum of 4 per annotator
    df = df.groupby("annotator_id").filter(lambda x: len(x) >= 4)

    # get user demographics
    dfs = []
    for annotator_id in df["annotator_id"].unique():
        annotator_df = df[df["annotator_id"] == annotator_id]
        user_demographics = row_to_user_demographics(annotator_df.iloc[0])
        # do a generic multivariate
        user_data = GenericMultivariate(
            df=annotator_df,
            descriptions=[
                "The following are user replies to a language model.\nUser demographics: "
                + user_demographics
            ],
            given_variables=["prompt", "model_response"],
            gen_variables=["reply"],
            name="community_alignment_reply",
        ).generate_many(**args)
        dfs.append(user_data)
    data = pd.concat(dfs)
    return data


def generate_community_alignment_individual_preferences(**kwargs):
    """Generate data loader for first turn conversations only"""
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=25),
        "max_inds": 1000,
    }
    args.update(kwargs)

    df = load_community_alignment()

    # drop so we have a minimum of 4 per annotator
    df = df.groupby("annotator_id").filter(lambda x: len(x) >= 4)

    # prompt is first_turn_prompt
    df["prompt"] = df["first_turn_prompt"]
    # response_a is first_turn_response_a
    df["response_a"] = df["first_turn_response_a"]
    # response_b is first_turn_response_b
    df["response_b"] = df["first_turn_response_b"]
    # response_c is first_turn_response_c
    df["response_c"] = df["first_turn_response_c"]
    # response_d is first_turn_response_d
    df["response_d"] = df["first_turn_response_d"]
    df["preferred_response"] = df["first_turn_preferred_response"]
    df["justification"] = df["first_turn_feedback"]

    # get user demographics
    dfs = []
    for annotator_id in df["annotator_id"].unique():
        annotator_df = df[df["annotator_id"] == annotator_id]
        user_demographics = row_to_user_demographics(annotator_df.iloc[0])
        # do a generic multivariate
        user_data = GenericMultivariate(
            df=annotator_df,
            descriptions=[
                "Given a prompt and a set of model responses, the user was tasked with selecting their most preferred response and an optional justification.\nUser demographics: "
                + user_demographics,
                "Given a prompt and a set of model responses, the user was tasked with selecting their most preferred response and an optional justification.",
            ],
            given_variables=[
                "prompt",
                "response_a",
                "response_b",
                "response_c",
                "response_d",
            ],
            gen_variables=["preferred_response", "justification"],
            name="community_alignment_reply",
        ).generate_many(**args)
        dfs.append(user_data)
    data = pd.concat(dfs)
    return data


# TODO - predict the model response maybe(?)
def generate_community_alignment_response(**kwargs):
    """Generate data loader for first turn conversations only"""
    args = {
        "seed": 42,
        "n_per": -1,
        # "n_iter": geom_and_poisson_iter(mean=25),
        "n_iter": 5,
    }
    args.update(kwargs)

    df = load_community_alignment()

    df = df.drop_duplicates(subset=["first_turn_prompt"])
    # make a list of all first 4 responses for each prompt
    df["responses"] = df.apply(
        lambda row: [
            row["first_turn_response_a"],
            row["first_turn_response_b"],
            row["first_turn_response_c"],
            row["first_turn_response_d"],
        ],
        axis=1,
    )
    # shuffle the responses
    # set random seed
    random.seed(42)

    def shuffle_list(x):
        shuffled = x.copy()
        random.shuffle(shuffled)
        return shuffled

    df["responses"] = df["responses"].apply(shuffle_list)
    # using jsonl turn to string
    df["responses"] = df["responses"].apply(lambda x: json.dumps(x))
    df["prompt"] = df["first_turn_prompt"]
    dfs = []
    for lang in df["assigned_lang"].unique():
        lang_df = df[df["assigned_lang"] == lang]
        # do a generic multivariate
        response_data = GenericMultivariate(
            df=lang_df,
            descriptions=[
                "You are a helpful assistant designed to help generate a list of 4 diverse responses for a user to consider."
            ],
            given_variables=["prompt"],
            gen_variables=["responses"],
            name="community_alignment_response",
        ).generate_many(**args)
        dfs.append(response_data)
    data = pd.concat(dfs)
    return data
