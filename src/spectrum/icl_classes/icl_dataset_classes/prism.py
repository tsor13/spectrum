import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_prism_prompts(**kwargs) -> pd.DataFrame:
    """
    Generate the prompts from the prism data.
    """
    conversations = pd.read_json(
        "hf://datasets/HannahRoseKirk/prism-alignment/conversations.jsonl", lines=True
    )

    args = {
        "seed": 42,
        "n_per": -1,  # Number of samples from the dataset
        "n_iter": geom_and_poisson_iter(
            mean=128
        ),  # randomly choose number of samples with mean ~128
    }
    args.update(kwargs)

    # These are the prompts from the original paper
    prompt_types = {
        "unguided": "Ask, request or talk to the model about anything. It is up to you!",
        "values guided": (
            "Ask, request or talk to the model about something important to you "
            "or that represents your values. This could be related to work, "
            "religion, family and relationship, politics or culture"
        ),
        "controversy guided": (
            "Ask, request or talk to the model about something controversial "
            "or where people would disagree in your community, culture or country."
        ),
    }
    conversations["conversation_type_description"] = conversations[
        "conversation_type"
    ].map(prompt_types)

    # strip opening_prompt
    conversations["opening_prompt"] = conversations["opening_prompt"].str.strip()

    generic_multivariate = GenericMultivariate(
        df=conversations,
        given_variables=["conversation_type_description"],
        gen_variables=["opening_prompt"],
        name="prism_prompt",
        descriptions=[
            "Generate a request to ask a language model, according to the conversation type description."
        ],
    )
    data = generic_multivariate.generate_many(**args)
    return data


def generate_prism_opening_prompts_individual(**kwargs) -> pd.DataFrame:
    """
    Generate opening prompts with demographic information as descriptions.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=96),
        "max_total": 800,
    }
    args.update(kwargs)

    # Load conversations and survey data
    conversations = pd.read_json(
        "hf://datasets/HannahRoseKirk/prism-alignment/conversations.jsonl", lines=True
    )
    survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")[
        "train"
    ].to_pandas()

    # Merge conversations with demographic data
    merged = conversations.merge(
        survey[
            [
                "user_id",
                "age",
                "gender",
                "education",
                "ethnicity",
                "religion",
                "location",
                "employment_status",
            ]
        ],
        on="user_id",
        how="inner",
    )

    # Extract simplified demographic information
    def extract_demo_info(row):
        demo_parts = []

        if pd.notna(row["age"]):
            demo_parts.append(f"Age: {row['age']}")

        if pd.notna(row["gender"]):
            demo_parts.append(f"Gender: {row['gender']}")

        if pd.notna(row["education"]):
            demo_parts.append(f"Education: {row['education']}")

        if pd.notna(row["employment_status"]):
            demo_parts.append(f"Employment: {row['employment_status']}")

        # Extract simplified location if available
        if pd.notna(row["location"]) and isinstance(row["location"], dict):
            if "reside_country" in row["location"]:
                demo_parts.append(f"Country: {row['location']['reside_country']}")

        # Extract simplified religion if available
        if pd.notna(row["religion"]) and isinstance(row["religion"], dict):
            if "simplified" in row["religion"]:
                demo_parts.append(f"Religion: {row['religion']['simplified']}")

        return "; ".join(demo_parts)

    merged["demographics"] = merged.apply(extract_demo_info, axis=1)

    # Map conversation types
    prompt_types = {
        "unguided": "Ask, request or talk to the model about anything. It is up to you!",
        "values guided": (
            "Ask, request or talk to the model about something important to you "
            "or that represents your values. This could be related to work, "
            "religion, family and relationship, politics or culture"
        ),
        "controversy guided": (
            "Ask, request or talk to the model about something controversial "
            "or where people would disagree in your community, culture or country."
        ),
    }
    merged["conversation_type_description"] = merged["conversation_type"].map(
        prompt_types
    )
    merged["opening_prompt"] = merged["opening_prompt"].str.strip()

    # Create descriptions that incorporate demographics
    def create_demographic_description(row):
        base_description = f"Generate an opening prompt for a conversation. {row['conversation_type_description']}"
        if row["demographics"]:
            return f"{base_description}\n\nUser demographics: {row['demographics']}"
        return base_description

    merged["demographic_description"] = merged.apply(
        create_demographic_description, axis=1
    )

    # loop through the user ids and create a new dataframe with the opening prompts for each user id
    dfs = []
    for user_id in merged["user_id"].unique():
        user_data = merged[merged["user_id"] == user_id]
        user_dataset = GenericMultivariate(
            df=user_data,
            given_variables=["conversation_type_description"],
            gen_variables=["opening_prompt"],
            name="prism_prompts_with_demographics",
            descriptions=[
                f"Generate opening prompts for conversations (all same individual)",
                f"Generate opening prompts for conversations given user demographics:\n\n{user_data['demographics'].iloc[0]} (all same individual)",
            ],
        )
        # increment seed by 1
        args["seed"] += 1
        dfs.append(user_dataset.generate_many(**args))
    data = pd.concat(dfs)

    return data


def generate_prism_individual_preferences(**kwargs) -> pd.DataFrame:
    """
    Generate individual preference predictions for first turn responses only,
    showing all 4 model responses labeled A-D with individual ratings in batch format.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=20),
        "max_inds": 500,
    }
    args.update(kwargs)

    # Load conversations and survey data
    conversations = pd.read_json(
        "hf://datasets/HannahRoseKirk/prism-alignment/conversations.jsonl", lines=True
    )
    survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")[
        "train"
    ].to_pandas()

    # Merge with demographic data
    merged = conversations.merge(
        survey[
            [
                "user_id",
                "age",
                "gender",
                "education",
                "ethnicity",
                "religion",
                "location",
                "employment_status",
            ]
        ],
        on="user_id",
        how="inner",
    )

    # Extract demographic information
    def extract_demo_info(row):
        demo_parts = []
        if pd.notna(row["age"]):
            demo_parts.append(f"Age: {row['age']}")
        if pd.notna(row["gender"]):
            demo_parts.append(f"Gender: {row['gender']}")
        if pd.notna(row["education"]):
            demo_parts.append(f"Education: {row['education']}")
        if pd.notna(row["employment_status"]):
            demo_parts.append(f"Employment: {row['employment_status']}")
        if pd.notna(row["location"]) and isinstance(row["location"], dict):
            if "reside_country" in row["location"]:
                demo_parts.append(f"Country: {row['location']['reside_country']}")
        if pd.notna(row["religion"]) and isinstance(row["religion"], dict):
            if "simplified" in row["religion"]:
                demo_parts.append(f"Religion: {row['religion']['simplified']}")
        return "; ".join(demo_parts)

    merged["demographics"] = merged.apply(extract_demo_info, axis=1)

    # Extract first turn model responses only - batch format
    preference_data = []
    for _, row in merged.iterrows():
        if not row["conversation_history"]:
            continue

        # Get the user's opening prompt
        opening_prompt = row["opening_prompt"]

        # Extract only first turn (turn 0) model responses
        first_turn_responses = []
        for turn_data in row["conversation_history"]:
            if (
                turn_data["role"] == "model"
                and turn_data.get("turn", -1) == 0
                and "score" in turn_data
                and "if_chosen" in turn_data
            ):
                first_turn_responses.append(turn_data)

        # Only process if we have multiple responses for the first turn
        if len(first_turn_responses) >= 2:
            # Sort by within_turn_id for consistent A-D labeling
            first_turn_responses.sort(key=lambda x: x.get("within_turn_id", 0))

            # Create model responses A, B, C, D
            model_responses = {}
            model_scores = {}
            chosen_response = None

            for i, response in enumerate(first_turn_responses):
                if i < 4:  # Only take first 4 responses
                    label = chr(65 + i)  # A, B, C, D
                    model_responses[f"response_{label.lower()}"] = response["content"]
                    model_scores[f"score_{label.lower()}"] = response[
                        "score"
                    ]  # Ensure integer
                    # model_scores[f'score_{label.lower()}'] = int(response['score'])  # Ensure integer
                    if response["if_chosen"]:
                        chosen_response = label.lower()  # a, b, c, or d

            # Only add if we have at least 2 responses
            if len(model_responses) >= 2:
                preference_data.append(
                    {
                        "user_id": row["user_id"],
                        "demographics": row["demographics"],
                        "conversation_type": row["conversation_type"],
                        "opening_prompt": opening_prompt,
                        # Model responses A-D
                        **model_responses,
                        # Individual scores for each model A-D
                        **model_scores,
                        # Single chosen response (a, b, c, or d)
                        "chosen_response": chosen_response,
                    }
                )

    preference_df = pd.DataFrame(preference_data)
    # drop na
    preference_df = preference_df.dropna()

    # make sure model_scores are all integers
    preference_df["score_a"] = preference_df["score_a"].astype(int)
    preference_df["score_b"] = preference_df["score_b"].astype(int)
    preference_df["score_c"] = preference_df["score_c"].astype(int)
    preference_df["score_d"] = preference_df["score_d"].astype(int)

    # Filter to users with sufficient preference data
    user_counts = preference_df["user_id"].value_counts()
    min_responses = 2  # Reduced since we're focusing on first turn only
    valid_users = user_counts[user_counts >= min_responses].index
    preference_df = preference_df[preference_df["user_id"].isin(valid_users)]

    # Create response and score variable lists dynamically based on available data
    response_vars = [
        col for col in preference_df.columns if col.startswith("response_")
    ]
    score_vars = [col for col in preference_df.columns if col.startswith("score_")]

    # Generate per-user preference models
    dfs = []
    for user_id in sorted(preference_df["user_id"].unique()):  # Sort for determinism
        user_data = preference_df[preference_df["user_id"] == user_id]
        demographics = user_data.iloc[0]["demographics"]

        given_vars = ["opening_prompt"] + response_vars
        gen_vars = ["chosen_response"] + score_vars

        user_prefs = IndividualMultivariate(
            df=user_data,
            individual_id_column="user_id",
            given_variables=given_vars,
            gen_variables=gen_vars,
            descriptions=[
                "The following is a prompt a user game to a chatbot, along with 4 responses. They were asked to choose their preferred response and rate the responses on a scale of 0 to 100, where 100 is the most preferred. The responses were labeled A, B, C, D.\nUser demographics: {demographics}"
            ],
            name="prism_individual_preferences",
        ).generate_many(**args)
        dfs.append(user_prefs)

    data = pd.concat(dfs) if dfs else pd.DataFrame()
    return data
