"""
uv run random_classes/dataset_loaders/lewidi_csc.py
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter


def load_csc_data(
    data_folder: str = "data/lewidi/CSC", split: str = "train"
) -> tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Load CSC sarcasm detection data and annotator metadata.

    Args:
        data_folder: Path to the CSC data folder
        split: Which split to load - 'train', 'dev', or 'both'

    Returns:
        tuple: (annotations_df, annotator_metadata)
    """
    # Load annotator metadata
    meta_path = os.path.join(data_folder, "CSC_annotators_meta.json")
    with open(meta_path, "r") as f:
        annotator_meta = json.load(f)

    # Load data based on split parameter
    all_annotations = []

    if split == "both":
        splits_to_load = ["train", "dev"]
    else:
        splits_to_load = [split]

    for split_name in splits_to_load:
        file_path = os.path.join(data_folder, f"CSC_{split_name}.json")
        with open(file_path, "r") as f:
            data = json.load(f)

        for instance_id, instance in data.items():
            # Extract text fields
            context = instance["text"]["context"]
            response = instance["text"]["response"]

            # Get annotator IDs and their annotations
            annotator_ids = instance["annotators"].split(",")
            annotations = instance["annotations"]

            # Get language code
            lang = instance.get("lang", "")

            # Create one row per annotator annotation
            for ann_id in annotator_ids:
                ann_id = ann_id.strip()  # Remove any whitespace
                if ann_id in annotations:
                    annotation_value = annotations[ann_id]

                    all_annotations.append(
                        {
                            "instance_id": instance_id,
                            "annotator_id": ann_id,
                            "context": context,
                            "response": response,
                            "lang": lang,
                            "sarcasm_rating": annotation_value,
                            "split": split_name,
                        }
                    )

    annotations_df = pd.DataFrame(all_annotations)
    return annotations_df, annotator_meta


def generate_csc_sarcasm_detection_individual(
    split: str = "train", **kwargs
) -> pd.DataFrame:
    """
    Generate CSC sarcasm detection annotations using GenericMultivariate per annotator.
    Each annotator's sarcasm detection patterns are modeled individually with demographic descriptions.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": 1000,
    }
    args.update(kwargs)

    # Load CSC data
    annotations_df, annotator_meta = load_csc_data(split=split)

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
            given_variables=["context", "response", "lang"],
            gen_variables=["sarcasm_rating"],
            # name="csc_sarcasm_detection",
            name=annotator_id,
            descriptions=[
                # "Rate the level of sarcasm in a conversation response based on the context.",
                f"Given a conversational context and response, rate how sarcastic the response is on a 1-6 scale.\nAnnotator demographics: {demographics}"
            ],
        )

        # Increment seed by 1 for each annotator
        args["seed"] += 1
        dfs.append(annotator_dataset.generate_many(**args))

    data = pd.concat(dfs)
    return data


def generate_csc_sarcasm_detection(split: str = "train", **kwargs) -> pd.DataFrame:
    """
    Generate CSC sarcasm detection annotations using GenericMultivariate.
    Treats all annotations as one dataset, ignoring individual annotator differences.
    """
    args = {
        "seed": 42,
        "n_per": -1,  # do all of the rows
        "n_iter": 1,  # do only ONE per generate (good for distributional w/out context)
    }
    args.update(kwargs)

    # Load CSC data
    annotations_df, annotator_meta = load_csc_data(split=split)

    # Create GenericMultivariate for all annotations together
    dataset = GenericMultivariate(
        df=annotations_df,
        given_variables=["context", "response", "lang"],
        gen_variables=["sarcasm_rating"],
        name="csc_sarcasm_detection",
        descriptions=[
            "Given a conversational context and response, rate how sarcastic the response is on a 1-6 scale."
        ],
    )

    data = dataset.generate_many(**args)
    return data
