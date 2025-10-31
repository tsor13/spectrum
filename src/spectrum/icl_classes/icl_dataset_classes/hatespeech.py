import json
import os
import sys

import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_hatespeech_raw(
    path: str = "data/hatespeech/classified_data_final_w_worker_hash.json",
) -> pd.DataFrame:
    lines = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {i}: {e}")
                return
    print("JSON file loaded successfully.")
    data_instances = []  # instance_id, rater_id, comment, toxic_score
    rater_metadata = []  # rater_id, demographics
    for line in tqdm(lines):
        comment_id = line["comment_id"]
        comment = line["comment"]
        for rating in line["ratings"]:
            rater_metadata_dict = {
                key: rating[key]
                for key in rating
                if key
                not in [
                    "toxic_score",
                    "is_profane",
                    "is_threat",
                    "is_identity_attack",
                    "is_insult",
                    "is_sexual_harassment",
                    "fine_to_see_online",
                    "remove_from_online",
                ]
            }
            # rater_id is a hash of rater_metadata_dict
            rater_id = hash(
                frozenset(rater_metadata_dict.items())
            )  # doing this because some workerIds seem to have multiple "sets" of metadata
            rater_metadata_dict["rater_id"] = rater_id
            rater_metadata.append(rater_metadata_dict)

            toxic_score = rating["toxic_score"]
            data_instances.append(
                {
                    "instance_id": comment_id,
                    "rater_id": rater_id,
                    "comment": comment,
                    "toxic_score": toxic_score,
                }
            )
    data_instances_df = pd.DataFrame(data_instances)
    return data_instances_df


def generate_hatespeech_individual_data(
    folder: str | None = None,
    # seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Predict the ratings of hate speech for an individual.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=128),
        "max_inds": 1000,
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/hatespeech"
    df = load_hatespeech_raw(
        path=os.path.join(folder, "classified_data_final_w_worker_hash.json")
    )
    # toxic score to string
    df["toxic_score"] = df["toxic_score"].apply(str)
    # make individual multivariate
    individual_multivariate = IndividualMultivariate(
        df=df,
        individual_id_column="rater_id",
        given_variables=["comment"],
        gen_variables=["toxic_score"],
        descriptions=[
            "Given a comment, predict the toxicity score from 0 to 4 (0 is not toxic, 4 is very toxic)."
        ],
        name="hatespeech_individual",
    )
    data = individual_multivariate.generate_many(**args)
    return data
