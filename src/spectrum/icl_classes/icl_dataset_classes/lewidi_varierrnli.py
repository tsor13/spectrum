"""
uv run random_classes/dataset_loaders/lewidi_varierrnli.py
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter


def load_varierrnli_data(
    data_folder: str = "data/lewidi/VariErrNLI", split: str = "train"
) -> tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Load VariErrNLI natural language inference data and annotator metadata.

    Args:
        data_folder: Path to the VariErrNLI data directory
        split: Which split to load - 'train', 'dev', or 'both'

    Returns:
        tuple: (annotations_df, annotator_metadata)
    """
    # Load annotator metadata (handle potential JSON formatting issues)
    meta_path = os.path.join(data_folder, "VariErrNLI_annotators_meta.json")
    with open(meta_path, "r") as f:
        content = f.read()
        # Fix trailing commas that break JSON parsing
        import re

        content = re.sub(r",(\s*[}\]])", r"\1", content)
        annotator_meta = json.loads(content)

    # Load training data only (dev commented out for now)
    all_annotations = []

    if split == "both":
        splits_to_load = ["train", "dev"]
    else:
        splits_to_load = [split]

    for split_name in splits_to_load:
        file_path = os.path.join(data_folder, f"VariErrNLI_{split_name}.json")
        with open(file_path, "r") as f:
            data = json.load(f)

        for instance_id, instance in data.items():
            # Extract text fields
            context = instance["text"]["context"]
            statement = instance["text"]["statement"]

            # Get annotator IDs and their annotations
            annotator_ids = instance["annotators"].split(",")
            annotations = instance["annotations"]

            # Get explanations if available
            explanations = instance.get("other_info", {}).get("explanations", [])

            # Get language code
            lang = instance.get("lang", "")

            # Create one row per annotator annotation
            for i, ann_id in enumerate(annotator_ids):
                ann_id = ann_id.strip()  # Remove any whitespace
                if ann_id in annotations:
                    annotation_value = annotations[ann_id]

                    # Get corresponding explanation for this annotator
                    explanation = explanations[i] if i < len(explanations) else ""

                    all_annotations.append(
                        {
                            "instance_id": instance_id,
                            "annotator_id": ann_id,
                            "context": context,
                            "statement": statement,
                            "lang": lang,
                            "nli_label": annotation_value,
                            "explanation": explanation.strip(),
                            "split": split_name,
                        }
                    )

    annotations_df = pd.DataFrame(all_annotations)
    return annotations_df, annotator_meta


def generate_varierrnli_nli_detection_individual(
    split: str = "train", **kwargs
) -> pd.DataFrame:
    """
    Generate VariErrNLI natural language inference annotations using GenericMultivariate per annotator.
    Each annotator's NLI patterns are modeled individually with demographic descriptions.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": 30,
    }
    args.update(kwargs)

    # Load VariErrNLI data
    annotations_df, annotator_meta = load_varierrnli_data(split=split)

    # Add ALL demographic information to annotations
    def get_annotator_demographics(annotator_id):
        """Extract ALL demographic info for an annotator."""
        if annotator_id in annotator_meta:
            meta = annotator_meta[annotator_id]
            meta["annotator_id"] = annotator_id
            demo_parts = []

            # Include ALL available metadata fields
            for field, value in meta.items():
                if value:  # Only include non-empty values
                    demo_parts.append(f"{field}: {value}")

            return "; ".join(demo_parts)
        return ""

    annotations_df["demographics"] = annotations_df["annotator_id"].apply(
        get_annotator_demographics
    )

    # Loop through annotator IDs and create GenericMultivariate for each
    dfs = []
    for annotator_id in annotations_df["annotator_id"].unique():
        annotator_data = annotations_df[annotations_df["annotator_id"] == annotator_id]

        # Get demographics for this annotator
        demographics = annotator_data["demographics"].iloc[0]

        annotator_dataset = GenericMultivariate(
            df=annotator_data,
            given_variables=["context", "statement", "lang"],
            gen_variables=["nli_label", "explanation"],
            # name="varierrnli_nli_detection",
            name=annotator_id,
            descriptions=[
                f"Given a premise and hypothesis from MNLI corpus, assign one or more labels from {{Entailment, Neutral, Contradiction}} indicating the logical relationship between them, and provide an explanation for your reasoning.\nAnnotator demographics: {demographics}"
            ],
        )

        # Increment seed by 1 for each annotator
        args["seed"] += 1
        dfs.append(annotator_dataset.generate_many(**args))

    data = pd.concat(dfs)
    return data


def generate_varierrnli_nli_detection_individual_categorical(
    split: str = "train", **kwargs
) -> pd.DataFrame:
    """
    Generate VariErrNLI natural language inference annotations using GenericMultivariate per annotator.
    Each annotator's NLI patterns are modeled individually with demographic descriptions.
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": 30,
    }
    args.update(kwargs)

    # Load VariErrNLI data
    annotations_df, annotator_meta = load_varierrnli_data(split=split)

    # Add ALL demographic information to annotations
    def get_annotator_demographics(annotator_id):
        """Extract ALL demographic info for an annotator."""
        if annotator_id in annotator_meta:
            meta = annotator_meta[annotator_id]
            meta["annotator_id"] = annotator_id
            demo_parts = []

            # Include ALL available metadata fields
            for field, value in meta.items():
                if value:  # Only include non-empty values
                    demo_parts.append(f"{field}: {value}")

            return "; ".join(demo_parts)
        return ""

    annotations_df["demographics"] = annotations_df["annotator_id"].apply(
        get_annotator_demographics
    )

    # Loop through annotator IDs and create GenericMultivariate for each
    dfs = []
    for annotator_id in annotations_df["annotator_id"].unique():
        annotator_data = annotations_df[annotations_df["annotator_id"] == annotator_id]

        # Get demographics for this annotator
        demographics = annotator_data["demographics"].iloc[0]

        annotator_dataset = GenericMultivariate(
            df=annotator_data,
            given_variables=["context", "statement", "lang"],
            # gen_variables=["nli_label", "explanation"],
            gen_variables=["nli_label"],
            # name="varierrnli_nli_detection",
            name=annotator_id,
            descriptions=[
                f"Given a premise and hypothesis from MNLI corpus, assign one or more labels from {{entailment, neutral, contradiction}} indicating the logical relationship between them.\nAnnotator demographics: {demographics}"
            ],
        )

        # Increment seed by 1 for each annotator
        args["seed"] += 1
        dfs.append(annotator_dataset.generate_many(**args))

    data = pd.concat(dfs)
    return data
