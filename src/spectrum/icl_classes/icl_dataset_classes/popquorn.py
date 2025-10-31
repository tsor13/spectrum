import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_popquorn_data(task_name: str, folder_path: str | None = None) -> pd.DataFrame:
    """
    Load the popquorn data from the folder path.
    """
    if folder_path is None:
        folder_path = "data/Potato-Prolific-Dataset/dataset"
    # [tsor13@klone-login01 bayesbench]$ ls data/Potato-Prolific-Dataset/dataset
    # email_rewriting  offensiveness  politeness_rating  question_answering
    # [tsor13@klone-login01 bayesbench]$ ls data/Potato-Prolific-Dataset/dataset/email_rewriting/
    # raw_data.csv
    path = os.path.join(folder_path, task_name, "raw_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    return pd.read_csv(path)


def generate_popquorn_individual(
    n_per_popquorn: int = 100, folder_path: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Generate the prompts from the popquorn data.
    """
    args = {
        "seed": 42,
        # 'n_per': 1,
        # 'n_iter': geom_and_poisson_iter(mean=128),
    }
    args.update(kwargs)
    datasets = [
        "email_rewriting",
        "offensiveness",
        "politeness_rating",
        "question_answering",
    ]
    # for dataset in datasets:
    #     data = load_popquorn_data(dataset, folder_path)
    #     breakpoint()
    # load email
    dfs = []
    email_data = load_popquorn_data("email_rewriting", folder_path)
    # strip original email and revised email of whitespeace
    email_data["original email"] = email_data["original email"].str.strip()
    email_data["revised email"] = email_data["revised email"].str.strip()
    # get user
    # get counts per user
    user_counts = email_data["user_id"].value_counts()
    # keep only with > 5
    user_counts = user_counts[user_counts > 5]
    # shuffle
    user_counts = user_counts.sample(frac=1, random_state=args["seed"])
    for user_id in user_counts.index[:n_per_popquorn]:
        user_data = email_data[email_data["user_id"] == user_id]
        # get gender, race, age, occcupation, education
        user_demographics = user_data[
            ["gender", "race", "age", "occupation", "education"]
        ].iloc[0]
        # to dict
        # get demographics
        user_demographics = str(user_demographics.to_dict())
        dfs.append(
            GenericMultivariate(
                df=user_data,
                given_variables=["original email"],
                gen_variables=["revised email"],
                name=f"popquorn_email_rewriting",
                descriptions=[
                    "Your collegue is trying to draft an email and is asking for your help. Please help to rewrite the message to make it more polite and professional.",
                    "The following are rewrites to make the email more polite and professional, all from the same individual.",
                    "The following are rewrites to make the email more polite and professional, all from the same individual.\nIndividual demographics: "
                    + user_demographics,
                ],
            ).generate_many(**args)
        )

    # load offensiveness
    offensiveness_data = load_popquorn_data("offensiveness", folder_path)

    # strip offensiveness
    offensiveness_data["text"] = offensiveness_data["text"].str.strip()
    # rename text to text_to_rate
    offensiveness_data.rename(columns={"text": "text_to_rate"}, inplace=True)
    # change offensiveness to str(int(
    offensiveness_data["offensiveness"] = offensiveness_data["offensiveness"].apply(
        lambda x: str(int(x))
    )
    # get counts per user
    user_counts = offensiveness_data["user_id"].value_counts()
    # keep only with > 5
    user_counts = user_counts[user_counts > 5].sample(
        frac=1, random_state=args["seed"] + 1
    )
    for user_id in user_counts.index[:n_per_popquorn]:
        user_data = offensiveness_data[offensiveness_data["user_id"] == user_id]
        # get gender, race, age, occcupation, education
        user_demographics = user_data[
            ["gender", "race", "age", "occupation", "education"]
        ].iloc[0]
        # to dict
        user_demographics = str(user_demographics.to_dict())
        dfs.append(
            GenericMultivariate(
                df=user_data,
                given_variables=["text_to_rate"],
                gen_variables=["offensiveness"],
                name=f"popquorn_offensiveness",
                descriptions=[
                    "Consider you read the above comment on Reddit, how offensive do you think it is? (1-5)",
                    "The following is a message that a user read. Please rate the message on a scale of 1 to 5, where 1 is not offensive and 5 is very offensive.\nIndividual demographics: "
                    + user_demographics,
                ],
            ).generate_many(**args)
        )

    # load politeness
    politeness_data = load_popquorn_data("politeness_rating", folder_path)
    # strip politeness
    politeness_data["text"] = politeness_data["text"].str.strip()
    # rename text to text_to_rate
    politeness_data.rename(columns={"text": "text_to_rate"}, inplace=True)
    # change politeness to str(int(
    politeness_data["politeness"] = politeness_data["politeness"].apply(
        lambda x: str(int(x))
    )
    # get counts per user
    user_counts = politeness_data["user_id"].value_counts()
    # keep only with > 5
    user_counts = user_counts[user_counts > 5].sample(
        frac=1, random_state=args["seed"] + 2
    )
    for user_id in user_counts.index[:n_per_popquorn]:
        user_data = politeness_data[politeness_data["user_id"] == user_id]
        # get gender, race, age, occcupation, education
        user_demographics = user_data[
            ["gender", "race", "age", "occupation", "education"]
        ].iloc[0]
        # to dict
        user_demographics = str(user_demographics.to_dict())
        dfs.append(
            GenericMultivariate(
                df=user_data,
                given_variables=["text_to_rate"],
                gen_variables=["politeness"],
                name=f"popquorn_politeness",
                descriptions=[
                    'Consider you read this email from a colleague, how polite do you think it is? (1-5), where 1 is "Not polite at all" and 5 is "Very polite"',
                    "The following is an email that a user read. Please rate the email on a scale of 1 to 5, where 1 is not polite and 5 is very polite.\nIndividual demographics: "
                    + user_demographics,
                ],
            ).generate_many(**args)
        )

    # load question answering
    question_answering_data = load_popquorn_data("question_answering", folder_path)

    # strip question answering
    question_answering_data["text"] = question_answering_data["text"].str.strip()
    # rename question to question_to_answer
    question_answering_data.rename(columns={"text": "passage"}, inplace=True)
    # because user_id is empty, need to use demographics as proxy
    question_answering_data["user_id"] = (
        question_answering_data["gender"]
        + "_"
        + question_answering_data["race"]
        + "_"
        + question_answering_data["age"]
        + "_"
        + question_answering_data["occupation"]
        + "_"
        + question_answering_data["education"]
    )
    # change difficulty to str(int(
    question_answering_data["difficulty"] = question_answering_data["difficulty"].apply(
        lambda x: str(int(x))
    )
    # get counts per user
    user_counts = question_answering_data["user_id"].value_counts()
    # keep only with > 5
    user_counts = user_counts[user_counts > 5].sample(
        frac=1, random_state=args["seed"] + 3
    )
    for user_id in user_counts.index[:n_per_popquorn]:
        user_data = question_answering_data[
            question_answering_data["user_id"] == user_id
        ]
        # get gender, race, age, occcupation, education
        user_demographics = user_data[
            ["gender", "race", "age", "occupation", "education"]
        ].iloc[0]
        # to dict
        user_demographics = str(user_demographics.to_dict())
        dfs.append(
            GenericMultivariate(
                df=user_data,
                given_variables=["passage", "question"],
                gen_variables=["groundtruth", "answer", "difficulty"],
                name=f"popquorn_question_answering",
                descriptions=[
                    'Answer the question by copying the answer from the passage, or respond with "No answer" if the question is not answered in the passage.\nBased on the passage and the question, give first the ground truth answer, then the answer that the user gave, and finally the user\'s difficulty rating of the question. (1-5)\nIndividual demographics: '
                    + user_demographics,
                    'Answer the question by copying the answer from the passage, or respond with "No answer" if the question is not answered in the passage.\nBased on the passage and the question, give first the ground truth answer, then the answer that the user gave, and finally the user\'s difficulty rating of the question. (1-5)',
                ],
            ).generate_many(**args)
        )

    # combine all dfs
    df = pd.concat(dfs)
    return df


def generate_popquorn_og_categorical(
    folder_path: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Generate the prompts from the popquorn data.
    """
    args = {
        "seed": 42,
        "n_per": 20,
        "n_iter": geom_and_poisson_iter(mean=128),
    }
    args.update(kwargs)
    datasets = [
        "email_rewriting",
        "offensiveness",
        "politeness_rating",
        "question_answering",
    ]
    dfs = []
    for dataset in datasets:
        data = load_popquorn_data(dataset, folder_path)
        # rename text to sample
        if "text" in data.columns:
            data = data.rename(columns={"text": "sample"})
        elif "original email" in data.columns:
            data = data.rename(columns={"original email": "sample"})
        # strip sample
        data["sample"] = data["sample"].str.strip()
        dfs.append(
            SingleVariableIID(
                categories=data["sample"].unique().tolist(),
                name=f"popquorn_{dataset}_texts",
                descriptions=[f"Data to use as samples for this task: {dataset}"],
                replacement=False,
            ).generate_many(**args)
        )

    df = pd.concat(dfs)
    return df
