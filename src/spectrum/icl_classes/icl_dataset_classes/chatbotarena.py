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


def load_chatbotarena(min_user_count=4):
    df = pd.read_parquet(
        "hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet"
    )
    # get user_id counts, throw out any with less than min_user_id_count
    user_id_counts = df["judge"].value_counts()
    df = df[df["judge"].isin(user_id_counts[user_id_counts >= min_user_count].index)]
    df["opening_prompt"] = (
        df["conversation_a"].apply(lambda x: x[0]["content"]).astype(str).str.strip()
    )
    df["model_a_response"] = (
        df["conversation_a"].apply(lambda x: x[1]["content"]).astype(str).str.strip()
    )
    df["model_b_response"] = (
        df["conversation_b"].apply(lambda x: x[1]["content"]).astype(str).str.strip()
    )
    # model_a          6139
    # model_b          5935
    # tie (bothbad)    3886
    # tie              1957
    # map to a, b, tie, bothbad
    df["winner"] = df["winner"].astype(str)
    df["winner"] = df["winner"].map(
        {
            "model_a": "a",
            "model_b": "b",
            "tie": "tie",
            "tie (bothbad)": "bothbad",
        }
    )
    return df


def generate_chatbotarena_prompts(**kwargs):
    args = {
        "seed": 42,
        "n_iter": 100,
        "max_inds": 1000,
    }
    args.update(kwargs)
    df = load_chatbotarena()
    # Index(['question_id', 'model_a', 'model_b', 'winner', 'judge',
    #    'conversation_a', 'conversation_b', 'turn', 'anony', 'language',
    #    'tstamp', 'openai_moderation', 'toxic_chat_tag'],
    # do categorical for opening prompt
    user_ids = df["judge"].unique()
    # shuffle user_ids
    np.random.seed(args["seed"])
    np.random.shuffle(user_ids)
    # take first n_users
    user_ids = user_ids[: args["max_inds"]]
    # do categorical for opening prompt
    dfs = []
    for user_id in user_ids:
        df_user = df[df["judge"] == user_id]
        # do categorical for opening prompt
        dfs.append(
            SingleVariableIID(
                df_user["opening_prompt"].tolist(),
                name=f"chatbotarena_opening_prompt",
                descriptions=["Language model prompt"],
                replacement=False,
            ).generate_many(**args)
        )
    return pd.concat(dfs)


def generate_chatbotarena_individual_prefs(**kwargs):
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=16),
        "max_inds": 1000,
    }
    args.update(kwargs)
    df = load_chatbotarena()
    data = IndividualMultivariate(
        df[
            [
                "opening_prompt",
                "model_a_response",
                "model_b_response",
                "winner",
                "judge",
            ]
        ],
        individual_id_column="judge",
        given_variables=["opening_prompt", "model_a_response", "model_b_response"],
        gen_variables=["winner"],
        descriptions=[
            f"The following are preferences for a single individual. They will prompt a language model (opening_prompt), and then will compare the two responses (model_a_response and model_b_response). The user will then rate the winner, with one of the following options: {df.winner.unique()}"
        ],
        name="chatbotarena_individual_prefs",
    ).generate_many(**args)
    return data


def generate_chatbotarena_assistant(**kwargs):
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 1000,
    }
    args.update(kwargs)
    df = load_chatbotarena()
    # shuffle data with seed
    df = df.sample(frac=1, random_state=args["seed"]).reset_index(drop=True)
    dfs = []
    for model in df.model_a.unique():
        # get all rows where model_a is model
        model_a_df = df[df.model_a == model]
        # get all rows where model_b is model
        model_b_df = df[df.model_b == model]
        # get opening prompt and responses into df
        combined_df = pd.DataFrame(
            {
                "opening_prompt": model_a_df["opening_prompt"].tolist()
                + model_b_df["opening_prompt"].tolist(),
                "response": model_a_df["model_a_response"].tolist()
                + model_b_df["model_b_response"].tolist(),
            }
        )
        # generic multivariate
        dfs.append(
            GenericMultivariate(
                combined_df[["opening_prompt", "response"]],
                given_variables=["opening_prompt"],
                gen_variables=["response"],
                name=f"chatbotarena_icl_assistance_{model}",
                descriptions=[
                    "You will be given opening_prompts to a language model (could be anywhere from pretty good to really good). Your job is to predict what the language model will say in response to the opening prompt. All responses are from the same model.",
                    f"You will be given opening_prompts to a language model (could be anywhere from pretty good to really good). Your job is to predict what the language model will say in response to the opening prompt. All responses are from the same model.\nModel: {model}",
                ],
            ).generate_many(**args)
        )
    return pd.concat(dfs)


def generate_chatbotarena_statistics(**kwargs):
    args = {
        "seed": 42,
        # "n_per": 5,
        "n_per": -1,
        "max_total": 500,
        "n_iter": 1000,
    }
    args.update(kwargs)
    df = load_chatbotarena(min_user_count=1)
    # get judge counts
    judge_counts = df["judge"].value_counts()
    # to list of counts
    counts = judge_counts.tolist()
    # to string
    counts = [str(count) for count in counts]
    # do categorical
    data = SingleVariableIID(
        counts, name="chatbotarena_statistics", descriptions=[], replacement=False
    ).generate_many(**args)
    return data
