"""
Note: The original polis dataset implementation used for training Spectrum models suffered from non-deterministic behavior - this one is rejiggered to be deterministic, but as a result, has a slightly different set of comments and votes.
"""

import glob
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID

polis_root = "data/openData/"
# 15-per-hour-seattle                    london.youth.policing
# american-assembly.bowling-green        march-on.operation-marchin-orders
# austria-climate.2vkxcncppn.2022-07-07  README.md
# austria-climate.5twd2jsnkf.2022-08-08  scoop-hivemind.affordable-housing
# austria-climate.5tzfrp5eaa.2022-07-07  scoop-hivemind.biodiversity
# austria-climate.7z7ejpbmv5.2022-08-08  scoop-hivemind.freshwater
# austria-climate.9xnndurbfm.2022-07-07  scoop-hivemind.taxes
# bg2050-volunteers                      scoop-hivemind.ubi
# brexit-consensus                       ssis.land-bank-farmland.2rumnecbeh.2021-08-01
# canadian-electoral-reform              vtaiwan.uberx
# football-concussions
polis_names = [
    "15-per-hour-seattle",
    "american-assembly.bowling-green",
    "austria-climate.2vkxcncppn.2022-07-07",
    "austria-climate.5twd2jsnkf.2022-08-08",
    "austria-climate.5tzfrp5eaa.2022-07-07",
    "austria-climate.7z7ejpbmv5.2022-08-08",
    "austria-climate.9xnndurbfm.2022-07-07",
    "bg2050-volunteers",
    "brexit-consensus",
    "canadian-electoral-reform",
    "football-concussions",
    "london.youth.policing",
    "march-on.operation-marchin-orders",
    "scoop-hivemind.affordable-housing",
    "scoop-hivemind.biodiversity",
    "scoop-hivemind.freshwater",
    "scoop-hivemind.taxes",
    "scoop-hivemind.ubi",
    "ssis.land-bank-farmland.2rumnecbeh.2021-08-01",
    "vtaiwan.uberx",
]

polis_folders = [os.path.join(polis_root, name) for name in polis_names]


def get_polis_dir(polis_name: str | int) -> str:
    if isinstance(polis_name, int):
        return polis_folders[polis_name]
    else:
        return os.path.join(polis_root, polis_name)


def load_polis_data(folder, drop_files=[]):
    # read in summary, stats-history, votes, comments, and participants-votes
    files = glob.glob(f"{folder}/*.csv")
    if len(files) == 0:
        raise ValueError(f"No files found in {folder}")
    file_map = {}
    # if not load_votes:
    #     # remove all files with "votes" in the name
    #     files = [file for file in files if "votes" not in file]
    # # remove stats-history
    # files = [file for file in files if "stats-history" not in file]
    for f in drop_files:
        files = [file for file in files if f not in file]
    for file in files:
        file_name = file.split("/")[-1].split(".")[0]
        # if summary, no header
        if file_name == "summary":
            file_map[file_name] = pd.read_csv(file, header=None)
        else:
            file_map[file_name] = pd.read_csv(file)
    # in comments drop where comment-body is na
    file_map["comments"] = file_map["comments"][
        file_map["comments"]["comment-body"].notna()
    ]

    comments_to_drop = []

    # add any comments that have test in it and fewer than 3 words
    comments_to_drop.extend(
        file_map["comments"][
            (
                file_map["comments"]["comment-body"].str.contains(
                    "test", case=False, na=False
                )
            )
            & (file_map["comments"]["comment-body"].str.split().str.len() < 3)
        ]["comment-id"].tolist()
    )

    if not "vtaiwan.uberx" in folder:
        # drop any comments with fewer than 4 words
        comments_to_drop.extend(
            file_map["comments"][
                (file_map["comments"]["comment-body"].str.split().str.len() < 4)
            ]["comment-id"].tolist()
        )

    # drop any na
    comments_to_drop.extend(
        file_map["comments"][file_map["comments"]["comment-body"].isna()][
            "comment-id"
        ].tolist()
    )

    print(f"Dropping {len(comments_to_drop)} comments")

    # drop comments by comment-id
    file_map["comments"] = file_map["comments"][
        ~file_map["comments"]["comment-id"].isin(comments_to_drop)
    ]
    # if votes, drop any votes that are not in the comments
    if "votes" in file_map:
        file_map["votes"] = file_map["votes"][
            file_map["votes"]["comment-id"].isin(file_map["comments"]["comment-id"])
        ]
    print(file_map["comments"].head())

    return file_map


def get_metadata(polis_data):
    summary_data = polis_data["summary"]
    # flip so [0] is column and [1] is entry
    summary_data = summary_data.T
    # make first row the header
    summary_data.columns = summary_data.iloc[0]
    summary_data = summary_data.iloc[1:]
    # summary_data['topic'].iloc[0]
    topic = summary_data["topic"].iloc[0]
    # dataset.ratings
    description = summary_data["conversation-description"].iloc[0]

    # prefix = f"""The following are votes from an online digital town hall.\n""" + f"Topic: {topic}\nDescription: {description}\nGiven each comment, people voted with either Agree, Disagree, or Pass.\n"
    metadata = {
        "topic": topic,
        "description": description,
    }
    keys_to_drop = []
    for key in metadata:
        if metadata[key] != metadata[key]:
            keys_to_drop.append(key)
    for key in keys_to_drop:
        del metadata[key]
    return metadata


def generate_polis_comment_data(
    polis_name_list: List[Union[str, int]] = None, **kwargs
) -> pd.DataFrame:
    """
    Generate polis comment data using SingleVariableIID.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=64),
        "max_inds": 500,
    }
    args.update(kwargs)

    if polis_name_list is None:
        polis_name_list = list(range(len(polis_names)))

    all_data = []

    for polis_name in tqdm(polis_name_list):
        folder = get_polis_dir(polis_name)
        polis_data = load_polis_data(
            folder, drop_files=["participants-votes", "stats-history"]
        )
        metadata = get_metadata(polis_data)
        comments = polis_data["comments"]
        comment_texts = comments["comment-body"].dropna().tolist()

        prefixes = [""]
        if "topic" in metadata:
            prefixes.append(f"Topic: {metadata['topic']}")
        if "description" in metadata:
            prefixes.append(f"Description: {metadata['description']}")
        if "topic" in metadata and "description" in metadata:
            prefixes.append(
                f"Topic: {metadata['topic']}\nDescription: {metadata['description']}"
            )
            prefixes.append(
                f"Description: {metadata['description']}\nTopic: {metadata['topic']}"
            )
        descriptors = []
        descriptors.append(f"The following are proposed comments.")
        descriptors.append(f"The following are comments from a digital town hall.")
        descriptors.append(f"The following are proposed statements.")
        descriptors.append(f"There will be statements for people to vote on.")
        descriptors.append(f"Comments ranging from a phrase up to a full paragraph.")
        descriptors.append(
            f"Opinion statements ranging from a phrase up to a full paragraph."
        )
        descriptors.append(f"Inputs from users in a digital town hall.")
        descriptions = []
        for descriptor in descriptors:
            for prefix in prefixes:
                descriptions.append(f"{prefix}\n{descriptor}".strip())
                descriptions.append(f"{descriptor}\n{prefix}".strip())
        # Create SingleVariableIID for comments
        comment_generator = SingleVariableIID(
            categories=comment_texts,
            name="polis_comments",
            descriptions=descriptions,
            replacement=False,
        )
        this_data = comment_generator.generate_many(**args)
        all_data.append(this_data)

    data = pd.concat(all_data, ignore_index=True)
    return data


def generate_polis_vote_data(
    polis_name_list: List[Union[str, int]] = None, **kwargs
) -> pd.DataFrame:
    """
    Generate polis voting data using IndividualMultivariate.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        # "n_iter": geom_and_poisson_iter(mean=32),
        # "max_inds": 1000,
        "max_inds": 500,
    }
    args.update(kwargs)

    if polis_name_list is None:
        # polis_name_list = list(range(min(3, len(polis_names))))  # Use first 3 by default for speed
        polis_name_list = list(range(len(polis_names)))

    # all_vote_data = []
    all_data = []

    # for polis_name in polis_name_list:
    # tqdm
    for polis_name in tqdm(polis_name_list):
        folder = get_polis_dir(polis_name)
        # comments, votes = load_polis_data(polis_name)
        polis_data = load_polis_data(
            folder, drop_files=["participants-votes", "stats-history"]
        )

        metadata = get_metadata(polis_data)
        comments = polis_data["comments"]
        votes = polis_data["votes"]

        # Merge comments and votes
        merged = pd.merge(
            votes, comments, left_on="comment-id", right_on="comment-id", how="inner"
        )

        # Filter to only include valid votes (not null)
        merged = merged[merged["vote"].notna()]

        # Map vote values to readable strings
        vote_mapping = {-1: "Disagree", 0: "Pass", 1: "Agree"}
        merged["vote_text"] = merged["vote"].map(vote_mapping)

        # Add voter id and comment text
        merged["voter_id"] = merged["voter-id"].astype(str)
        merged["comment_text"] = merged["comment-body"]

        # Keep relevant columns
        vote_data = merged[["voter_id", "comment_text", "vote_text"]].copy()

        # get value counts for each voter
        min_votes = 4
        # drop all voters with less than min_votes
        voter_counts = vote_data["voter_id"].value_counts()
        drop_voters = voter_counts[voter_counts < min_votes].index.tolist()
        vote_data = vote_data[~vote_data["voter_id"].isin(drop_voters)]
        # vote_data['polis_name'] = polis_name
        # all_vote_data.append(vote_data)
        # get description from metadata
        descriptions = []

        prefixes = [""]
        if "topic" in metadata:
            prefixes.append(f"Topic: {metadata['topic']}")
        if "description" in metadata:
            prefixes.append(f"Description: {metadata['description']}")
        if "topic" in metadata and "description" in metadata:
            prefixes.append(
                f"Topic: {metadata['topic']}\nDescription: {metadata['description']}"
            )
            prefixes.append(
                f"Description: {metadata['description']}\nTopic: {metadata['topic']}"
            )
        descriptors = []
        descriptors.append(
            f"The following are votes, all from the same individual. Options: Agree, Disagree, Pass"
        )
        descriptors.append(
            f"The following are votes from a digital town hall.\nOptions: Agree, Disagree, Pass"
        )
        descriptors.append(
            f"Here are someone's votes on a digital town hall.\nOptions: Agree, Disagree, Pass"
        )
        descriptors.append(
            f"Here are the votes from someone on a digital town hall.\nPossible responses: Agree, Disagree, Pass"
        )

        for descriptor in descriptors:
            for prefix in prefixes:
                descriptions.append(f"{prefix}\n{descriptor}".strip())
                descriptions.append(f"{descriptor}\n{prefix}".strip())
        vote_generator = IndividualMultivariate(
            df=vote_data,
            individual_id_column="voter_id",
            given_variables=["comment_text"],
            gen_variables=["vote_text"],
            descriptions=descriptions,
            name="polis_individual_votes",
        )
        this_data = vote_generator.generate_many(**args)
        all_data.append(this_data)

    data = pd.concat(all_data, ignore_index=True)
    return data
