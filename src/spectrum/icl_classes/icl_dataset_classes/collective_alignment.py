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


def load_collective_alignment(sample_size=100_000):
    """Load the OpenAI Collective Alignment Dataset"""
    # Load comparisons subset for prompts and responses
    dataset = load_dataset(
        "openai/collective-alignment-1", "comparisons", split="train", streaming=True
    )

    samples = []
    for i, sample in enumerate(dataset):
        if i >= sample_size:
            break
        samples.append(sample)

    df = pd.DataFrame(samples)
    return df


def load_collective_alignment_annotators(sample_size=100_000):
    """Load the OpenAI Collective Alignment Dataset with annotator info"""
    # Load merged dataset for annotator demographics and feedback
    dataset = load_dataset(
        "openai/collective-alignment-1",
        "merged_comparison_annotators",
        split="train",
        streaming=True,
    )

    samples = []
    for i, sample in enumerate(dataset):
        if i >= sample_size:
            break
        samples.append(sample)

    df = pd.DataFrame(samples)
    return df


def extract_conversation_messages(prompt):
    """Extract conversation messages from prompt structure"""
    messages = prompt["messages"]
    conversation_parts = []

    for msg in messages:
        if msg["role"] == "user":
            # conversation_parts.append(f"User: {msg['content']}")
            conversation_parts.append(msg["content"])

    return "\n".join(conversation_parts)


def row_to_annotator_demographics(demographics_dict):
    """Convert annotator demographics to JSON string"""
    return json.dumps(demographics_dict)


def generate_collective_alignment_individual(**kwargs):
    """Generate data loader for rater feedback with demographics in context"""
    args = {
        "seed": 42,
        # "n_per": -1,
        # "n_iter": geom_and_poisson_iter(mean=20),
        # "max_inds": 500,
    }
    args.update(kwargs)

    # Load both datasets
    comparison_df = load_collective_alignment()
    annotator_df = load_collective_alignment_annotators()

    # Create lookup for prompt info
    prompt_lookup = {}
    for _, row in comparison_df.iterrows():
        prompt_id = row["prompt_id"]
        conversation = extract_conversation_messages(row["prompt"])
        responses = {}
        for response in row["responses"]:
            idx = response["response_index"]
            content = response["messages"][0]["content"]
            responses[f"response_{idx.lower()}"] = content

        prompt_lookup[prompt_id] = {
            "conversation": conversation,
            "responses": responses,
        }

    # Filter to annotators with sufficient data
    annotator_counts = annotator_df["annotator_id"].value_counts()
    min_responses = 3
    valid_annotators = annotator_counts[annotator_counts >= min_responses].index
    annotator_df = annotator_df[annotator_df["annotator_id"].isin(valid_annotators)]

    # Prepare individual feedback data
    individual_data = []
    for _, row in annotator_df.iterrows():
        if row["prompt_id"] in prompt_lookup:
            prompt_info = prompt_lookup[row["prompt_id"]]

            # Extract rankings and rationales
            rankings = row["ranking_blocks"]
            personal_list = rankings.get("personal", [])
            world_list = rankings.get("world", [])
            unacceptable_list = rankings.get("unacceptable", [])

            personal_ranking = (
                personal_list[0].get("ranking", "") if personal_list else ""
            )
            world_ranking = world_list[0].get("ranking", "") if world_list else ""
            personal_rationale = (
                personal_list[0].get("rationale", "") if personal_list else ""
            )
            world_rationale = world_list[0].get("rationale", "") if world_list else ""

            # Unacceptable responses
            unacceptable = (
                unacceptable_list[0].get("rating", []) if unacceptable_list else []
            )
            unacceptable_rationale = (
                unacceptable_list[0].get("rationale", "") if unacceptable_list else ""
            )

            individual_data.append(
                {
                    "annotator_id": row["annotator_id"],
                    "conversation": prompt_info["conversation"],
                    "response_a": prompt_info["responses"].get("response_a", ""),
                    "response_b": prompt_info["responses"].get("response_b", ""),
                    "response_c": prompt_info["responses"].get("response_c", ""),
                    "response_d": prompt_info["responses"].get("response_d", ""),
                    "importance": row["importance"],
                    "representativeness": row["representativeness"],
                    "subjectivity": row["subjectivity"],
                    "personal_ranking": personal_ranking,
                    "world_ranking": world_ranking,
                    "personal_rationale": personal_rationale,
                    "world_rationale": world_rationale,
                    "unacceptable_responses": "|".join(unacceptable),
                    "unacceptable_rationale": unacceptable_rationale,
                    "demographics": row_to_annotator_demographics(row["demographics"]),
                }
            )

    individual_df = pd.DataFrame(individual_data)

    # Generate per-annotator data with deterministic ordering
    dfs = []
    for annotator_id in sorted(individual_df["annotator_id"].unique()):
        annotator_df = individual_df[individual_df["annotator_id"] == annotator_id]
        demographics = annotator_df.iloc[0]["demographics"]

        user_data = GenericMultivariate(
            df=annotator_df,
            descriptions=[
                f"The following are AI response evaluations from a single individual with specific demographics and values. They were asked to provide their own personal ranking of the model responses, a ranking for what is best for the world, a rationale for each of them, which (if any) unacceptable responses are unacceptable, and a rationale for why they are unacceptable. Finally, they were asked to provide the importance of getting the correct answer, how representative the response is of something they would write, and whether they feel the evaluation is subjective.\nAnnotator demographics: {demographics}"
            ],
            given_variables=[
                "conversation",
                "response_a",
                "response_b",
                "response_c",
                "response_d",
            ],
            gen_variables=[
                "personal_ranking",
                "world_ranking",
                "personal_rationale",
                "world_rationale",
                "unacceptable_responses",
                "unacceptable_rationale",
                "importance",
                "representativeness",
                "subjectivity",
            ],
            name="collective_alignment_individual",
        ).generate_many(**args)
        dfs.append(user_data)

    data = pd.concat(dfs)
    return data
