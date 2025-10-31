"""
uv run random_classes/dataset_loaders/ambient.py
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_ambient_data():
    """Load AmbiEnt dataset from the JSONL file."""
    data_path = "data/ambient/AmbiEnt/ambient_full.jsonl"

    data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def load_ambient_linguist_annotations():
    """Load AmbiEnt linguist annotations and reformat with worker_id per row."""
    data_path = "data/ambient/AmbiEnt/linguist_annotations.jsonl"

    reformatted_data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())

            # Extract basic info
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            worker_ids = item["worker_ids"]
            annotations = item["annotations"]

            # Create one row per worker_id with their annotation
            # Assuming annotations come in same order as worker_ids
            for worker_id, annotation in zip(worker_ids, annotations):
                reformatted_data.append(
                    {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "worker_id": worker_id,
                        "annotation": annotation,
                    }
                )

    return pd.DataFrame(reformatted_data)


def load_ambient_crowdworker_annotations():
    """Load AmbiEnt crowdworker annotations from the JSONL file."""
    data_path = "data/ambient/AmbiEnt/analysis/crowdworker_annotations.jsonl"

    data = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def load_ambient_interpretation_labels():
    """Load and expand crowdworker data into interpretation-label pairs."""
    df = load_ambient_crowdworker_annotations()

    expanded_data = []
    for _, row in df.iterrows():
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        ambiguous_sent = row["ambiguous_sent"]
        interpretations = row["interpretations"]
        labels = row["labels"]

        # Create one row per interpretation with its corresponding label
        for interpretation, label in zip(interpretations, labels):
            if label is not None:  # Skip null labels
                expanded_data.append(
                    {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "ambiguous_sent": ambiguous_sent,
                        "interpretation": interpretation,
                        "entailment_label": label,
                    }
                )

    return pd.DataFrame(expanded_data)


def load_ambient_annotation_distributions():
    """Load and process crowdworker data to create annotation distributions."""
    df = load_ambient_crowdworker_annotations()

    distribution_data = []
    for _, row in df.iterrows():
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        ambiguous_sent = row["ambiguous_sent"]

        # Aggregate all q*_gold arrays to get overall distribution
        all_annotations = []
        for q_key in ["q0_gold", "q1_gold", "q2_gold", "q3_gold"]:
            if q_key in row:
                all_annotations.extend(
                    [label for label in row[q_key] if label is not None]
                )

        # Count occurrences of each label
        if all_annotations:
            from collections import Counter

            label_counts = Counter(all_annotations)

            # Create a formatted distribution string
            total = sum(label_counts.values())
            distribution_parts = []

            label_distribution = {}
            for label in ["entailment", "neutral", "contradiction"]:
                count = label_counts.get(label, 0)
                label_distribution[label + "_fraction"] = f"{count}/{total}"
                label_distribution[label + "_percentage"] = float(
                    np.round(count / total * 100, 4)
                )

            distribution_data.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "ambiguous_sent": ambiguous_sent,
                    "label_distribution": label_distribution,
                }
            )

    return pd.DataFrame(distribution_data)


def generate_ambient_premise_hypothesis(**kwargs) -> pd.DataFrame:
    """
    Generate ambiguous premise/hypothesis pairs from AmbiEnt dataset.
    Task 1: Generate the ambiguous premise and hypothesis pairs as shown.
    """
    df = load_ambient_data()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)

    # Generate ambiguous premise/hypothesis pairs
    generator = GenericMultivariate(
        df=df,
        name="ambient_premise_hypothesis",
        given_variables=[],
        gen_variables=["premise", "hypothesis"],
        descriptions=[
            "Generate ambiguous premise and hypothesis pairs for natural language inference tasks. These pairs contain linguistic ambiguities that can lead to multiple valid interpretations.",
            "Create premise-hypothesis pairs where the premise contains ambiguity that affects the entailment relationship with the hypothesis.",
            "Generate: premise (ambiguous statement), hypothesis (statement to evaluate against premise). Natural language inference with ambiguity.",
        ],
    )

    return generator.generate_many(**args)


def generate_ambient_ambiguity_detection(**kwargs) -> pd.DataFrame:
    """
    Generate ambiguity detection task from AmbiEnt dataset.
    Task 2: Given premise/hypothesis pair, predict whether premise/hypothesis is ambiguous.
    """
    df = load_ambient_data()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 1 to the seed to avoid the same seed for different tasks
    args["seed"] += 1

    # Convert boolean flags to strings for generation
    # df['premise_ambiguous_str'] = df['premise_ambiguous'].astype(str)
    # df['hypothesis_ambiguous_str'] = df['hypothesis_ambiguous'].astype(str)

    # Generate ambiguity detection predictions
    generator = GenericMultivariate(
        df=df,
        name="ambient_ambiguity_detection",
        given_variables=["premise", "hypothesis"],
        # gen_variables=["premise_ambiguous_str", "hypothesis_ambiguous_str"],
        gen_variables=["premise_ambiguous", "hypothesis_ambiguous"],
        descriptions=[
            "Given a premise and hypothesis pair, determine whether the premise is ambiguous and whether the hypothesis is ambiguous. Answer with 'True' or 'False' for each.",
            "Analyze premise-hypothesis pairs to detect linguistic ambiguity in either statement.",
            "Given: premise, hypothesis. Generate: premise_ambiguous_str (True/False), hypothesis_ambiguous_str (True/False). Ambiguity detection task.",
        ],
    )

    return generator.generate_many(**args)


def generate_ambient_disambiguation(**kwargs) -> pd.DataFrame:
    """
    Generate disambiguation task from AmbiEnt dataset.
    Task 3: Given ambiguous premise/hypothesis pair and desired label, rewrite to remove ambiguity.
    """
    df = load_ambient_data()

    # Expand disambiguations into separate rows
    disambiguation_rows = []
    for _, row in df.iterrows():
        for disambig in row["disambiguations"]:
            disambiguation_row = {
                "original_premise": row["premise"],
                "original_hypothesis": row["hypothesis"],
                "target_label": disambig["label"],
                "disambiguated_premise": disambig["premise"],
                "disambiguated_hypothesis": disambig["hypothesis"],
            }
            disambiguation_rows.append(disambiguation_row)

    disambig_df = pd.DataFrame(disambiguation_rows)

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 2 to the seed to avoid the same seed for different tasks
    args["seed"] += 2

    # Generate disambiguation rewrites
    generator = GenericMultivariate(
        df=disambig_df,
        name="ambient_disambiguation",
        given_variables=["original_premise", "original_hypothesis", "target_label"],
        gen_variables=["disambiguated_premise", "disambiguated_hypothesis"],
        descriptions=[
            "Given an ambiguous premise-hypothesis pair and a target entailment label, rewrite the premise and/or hypothesis to remove ambiguity and achieve the target label.",
            "Disambiguate premise-hypothesis pairs by rewriting them to clearly express a specific entailment relationship (entailment, neutral, or contradiction).",
            "Given: original_premise (ambiguous), original_hypothesis, target_label (entailment/neutral/contradiction). Generate: disambiguated_premise, disambiguated_hypothesis (unambiguous versions).",
        ],
    )

    return generator.generate_many(**args)


def generate_ambient_linguist_annotations(**kwargs) -> pd.DataFrame:
    """
    Generate linguist annotation predictions from AmbiEnt dataset.
    Task 4: Given premise/hypothesis pair, predict the annotation for each individual linguist.
    """
    df = load_ambient_linguist_annotations()

    args = {
        "seed": 42,
        "n_per": 2,  # 2 per individual, because only 28 linguists
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 3 to the seed to avoid the same seed for different tasks
    args["seed"] += 3

    # Generate linguist annotation predictions using IndividualMultivariate
    generator = IndividualMultivariate(
        df=df,
        individual_id_column="worker_id",
        given_variables=["premise", "hypothesis"],
        gen_variables=["annotation"],
        name="ambient_linguist_annotations",
        descriptions=[
            "Given a premise-hypothesis pair, predict the entailment annotation that this specific linguist would provide (entailment, neutral, contradiction, or combinations like 'entailment|neutral').",
            "Predict individual linguist annotations for natural language inference tasks based on premise-hypothesis pairs.",
            "Given: premise, hypothesis. Generate: annotation (entailment judgment by this individual linguist).",
        ],
    )

    return generator.generate_many(**args)


def generate_ambient_interpretation_labels(**kwargs) -> pd.DataFrame:
    """
    Generate interpretation-label predictions from AmbiEnt crowdworker dataset.
    Task 5: Given premise, hypothesis, ambiguous_sent, and interpretation, predict the entailment label.
    """
    df = load_ambient_interpretation_labels()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 4 to the seed to avoid the same seed for different tasks
    args["seed"] += 4

    # Generate interpretation-label predictions
    generator = GenericMultivariate(
        df=df,
        name="ambient_interpretation_labels",
        given_variables=["premise", "hypothesis", "ambiguous_sent", "interpretation"],
        gen_variables=["entailment_label"],
        descriptions=[
            "Given a premise-hypothesis pair with an ambiguous sentence and a specific interpretation of that ambiguity, predict the entailment label (entailment, neutral, or contradiction).",
            "Predict entailment labels based on specific interpretations of ambiguous premise-hypothesis pairs.",
            "Given: premise, hypothesis, ambiguous_sent (which sentence is ambiguous), interpretation (specific meaning). Generate: entailment_label (entailment/neutral/contradiction).",
        ],
    )

    return generator.generate_many(**args)


def generate_ambient_annotation_distributions(**kwargs) -> pd.DataFrame:
    """
    Generate annotation distribution predictions from AmbiEnt crowdworker dataset.
    Task 6: Given premise, hypothesis, and ambiguous_sent, predict the distribution of labels across annotators.
    """
    df = load_ambient_annotation_distributions()

    args = {
        "seed": 42,
        "n_per": 50,
        "n_iter": geom_and_poisson_iter(mean=64),
    }
    args.update(kwargs)
    # add 5 to the seed to avoid the same seed for different tasks
    args["seed"] += 5

    # Generate annotation distribution predictions
    generator = GenericMultivariate(
        df=df,
        name="ambient_annotation_distributions",
        given_variables=["premise", "hypothesis", "ambiguous_sent"],
        gen_variables=["label_distribution"],
        descriptions=[
            "Given an ambiguous premise-hypothesis pair, predict the distribution of entailment labels that human annotators would provide. Format as counts and proportions for each label type.",
            "Predict how human annotators would distribute their judgments across entailment, neutral, and contradiction labels for ambiguous premise-hypothesis pairs.",
            "Given: premise, hypothesis, ambiguous_sent (which is ambiguous). Generate: label_distribution (annotator consensus patterns).",
        ],
    )

    return generator.generate_many(**args)
