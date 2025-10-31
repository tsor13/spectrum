"""
uv run random_classes/dataset_loaders/lewidi_mp.py
"""

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter


def load_mp_data(
    data_folder: str = "data/lewidi/MP", split: str = "train"
) -> tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Load MP irony detection data and annotator metadata.

    Args:
        data_folder: Path to the MP data folder
        split: Which split to load ('train', 'dev', or 'both')

    Returns:
        tuple: (annotations_df, annotator_metadata)
    """
    # Load annotator metadata (handle potential JSON formatting issues)
    meta_path = os.path.join(data_folder, "MP_annotators_meta.json")
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
        file_path = os.path.join(data_folder, f"MP_{split_name}.json")
        with open(file_path, "r") as f:
            data = json.load(f)

        for instance_id, instance in data.items():
            # Extract text fields
            post = instance["text"]["post"]
            reply = instance["text"]["reply"]

            # Extract context variables from other_info
            other_info = instance.get("other_info", {})
            source = other_info.get("source", "")
            level = str(other_info.get("level", ""))
            language_variety = other_info.get("language_variety", "")

            # Get language code
            lang = instance.get("lang", "")

            # Get annotator IDs and their annotations
            annotator_ids = instance["annotators"].split(",")
            annotations = instance["annotations"]

            # Create one row per annotator annotation
            for ann_id in annotator_ids:
                ann_id = ann_id.strip()  # Remove any whitespace
                if ann_id in annotations:
                    annotation_value = annotations[ann_id]

                    all_annotations.append(
                        {
                            "instance_id": instance_id,
                            "annotator_id": ann_id,
                            "post": post,
                            "reply": reply,
                            "source": source,
                            "level": level,
                            "language_variety": language_variety,
                            "lang": lang,
                            "irony_rating": annotation_value,
                            "split": split_name,
                        }
                    )

    annotations_df = pd.DataFrame(all_annotations)
    return annotations_df, annotator_meta


def generate_mp_irony_detection_individual(
    split: str = "train", **kwargs
) -> pd.DataFrame:
    """
    Generate MP irony detection annotations using GenericMultivariate per annotator.
    Each annotator's irony detection patterns are modeled individually with demographic descriptions.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": 1000,
    }
    args.update(kwargs)

    # Load MP data
    annotations_df, annotator_meta = load_mp_data(split=split)

    # Add ALL demographic information to annotations
    def get_annotator_demographics(annotator_id):
        """Extract ALL demographic info for an annotator."""
        # remove Ann
        annotator_id = annotator_id.replace("Ann", "")
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
            given_variables=[
                "post",
                "reply",
                "source",
                "level",
                "language_variety",
                "lang",
            ],
            gen_variables=["irony_rating"],
            # name="mp_irony_detection",
            name=annotator_id,
            descriptions=[
                f"Given a post-reply pair from social media (Twitter/Reddit), determine whether the reply is ironic given the post. Context includes platform source, reply depth level, language variety, and language code. Binary irony detection task.\nAnnotator demographics: {demographics}"
            ],
        )

        # Increment seed by 1 for each annotator
        args["seed"] += 1
        dfs.append(annotator_dataset.generate_many(**args))

    data = pd.concat(dfs)
    return data


def generate_mp_irony_detection(split: str = "train", **kwargs) -> pd.DataFrame:
    """
    Generate MP irony detection annotations using GenericMultivariate.
    Treats all annotations as one dataset, ignoring individual annotator differences.
    """
    args = {
        "seed": 42,
        "n_per": -1,  # do all of the rows
        "n_iter": 1,  # do only ONE per generate (good for distributional w/out context)
    }
    args.update(kwargs)

    # Load MP data
    annotations_df, annotator_meta = load_mp_data(split=split)

    # Create GenericMultivariate for all annotations together
    dataset = GenericMultivariate(
        df=annotations_df,
        given_variables=[
            "post",
            "reply",
            "source",
            "level",
            "language_variety",
            "lang",
        ],
        gen_variables=["irony_rating"],
        name="mp_irony_detection",
        descriptions=[
            "Given a post-reply pair from social media (Twitter/Reddit), determine whether the reply is ironic given the post. Context includes platform source, reply depth level, language variety, and language code. Binary irony detection task."
        ],
    )

    data = dataset.generate_many(**args)
    return data
