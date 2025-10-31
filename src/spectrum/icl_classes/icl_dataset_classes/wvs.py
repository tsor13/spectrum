"""
uv run random_classes/dataset_loaders/wvs.py
"""

import os

# add parent directory
import sys

import pandas as pd
import pyreadstat
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate


def load_wvs_helper(folder: str | None = None) -> pd.DataFrame:
    if folder is None:
        folder = "data/wvs"
    # [tsor13@klone-login01 data]$ ls wvs/
    # WVS_Cross-National_Wave_7_csv_v6_0.csv  WVS_Cross-National_Wave_7_inverted_sav_v6_0.sav
    # read in sav file
    df, meta = pyreadstat.read_sav(
        os.path.join(folder, "WVS_Cross-National_Wave_7_inverted_sav_v6_0.sav")
    )
    descriptions = meta.column_names_to_labels
    # get mappings from value labels to values
    value_mappings = meta.variable_value_labels
    col = list(descriptions.keys())[100]
    # get all columns with Q in them
    q_cols = [k for k in df.columns if k.startswith("Q")]
    other_cols = [k for k in df.columns if not k.startswith("Q")]
    demographic_cols = [
        "B_COUNTRY",
        "N_TOWN",
        "S_INTLANGUAGE",
        # ... a whole bunch more if we want to get into it
    ]
    # value_mappings[col]
    # loop through rows
    return {
        "data": df,
        "descriptions": descriptions,
        "value_mappings": value_mappings,
        "q_cols": q_cols,
        "other_cols": other_cols,
        "demographic_cols": demographic_cols,
    }


def generate_wvs_individual(
    n_individuals: int = 2000, folder: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Generate a random individual from the WVS dataset.
    """
    args = {
        "seed": 42,
    }
    args.update(kwargs)
    out = load_wvs_helper(folder)
    data = out["data"]
    descriptions = out["descriptions"]
    value_mappings = out["value_mappings"]
    q_cols = out["q_cols"]
    other_cols = out["other_cols"]
    demographic_cols = out["demographic_cols"]
    # shuffle data
    data = data.sample(frac=1, random_state=args["seed"]).reset_index(drop=True)
    # loop through rows
    dfs = []
    for i in tqdm(range(n_individuals)):
        # get a random row
        row = data.iloc[i]
        # get demographic values
        demographics = {}
        for col in demographic_cols:
            # check if row[col] is na
            if pd.isna(row[col]):
                continue
            demographics[descriptions[col]] = value_mappings[col][row[col]]

        responses = []
        for col in q_cols:
            # check if row[col] is na
            if pd.isna(row[col]):
                continue
            try:
                # reverse because it's more natural
                options = list(value_mappings[col].values())[::-1]
                # if > 10 options, debug

                question_text = descriptions[col]

                if len(value_mappings[col]) > 17:
                    continue
                if col in ["Q261", "Q262", "Q270", "Q274"]:
                    continue
                else:
                    response = value_mappings[col][row[col]]
                # do alpha map from options to A/B/C/D/E/F/G...
                # option_map = {v: chr(65 + i) for i, v in enumerate(eval(options))}
                # letters = [chr(65 + i) for i in range(len(options))][::-1]
                option_to_letter_map = {v: chr(65 + i) for i, v in enumerate(options)}
                letter_to_option_map = {v: k for k, v in option_to_letter_map.items()}
                option_str = "\n".join(
                    [f"{k}: {v}" for k, v in letter_to_option_map.items()]
                )
                full_text = f"{question_text}\nOptions:\n{option_str}"
                response_letter = option_to_letter_map[response]
                responses.append(
                    {
                        "col": col,
                        "question": question_text,
                        "options": options,
                        "question_and_options": full_text,
                        "option_to_letter_map": option_to_letter_map,
                        "letter_to_option_map": letter_to_option_map,
                        "response_letter": response_letter,
                        "response": response,
                    }
                )
            except Exception as e:
                print(col)
                print(question_text)
                print(e)
                breakpoint()
        # to df
        ratings = pd.DataFrame(responses)
        # make a generic multivariate
        mv = GenericMultivariate(
            df=ratings,
            descriptions=[
                "response ~ question + options",
                # 'Responses from a global survey',
                "Responses from this user: " + str(demographics),
            ],
            # given_variables=['question', 'options'],
            # gen_variables=['response'],
            given_variables=["question_and_options"],
            gen_variables=["response_letter"],
            name="wvs",
        )
        # generate data
        args["seed"] = args["seed"] + 1
        dfs.append(mv.generate_many(**args))
    all_data = pd.concat(dfs)
    return all_data


if __name__ == "__main__":
    # df = load_wvs_helper()
    # data = generate_wvs_individual(n_individuals=100_000)
    # data = generate_wvs_individual(n_individuals=10_000, seed=2)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    data = generate_wvs_individual(
        n_individuals=100,
        seed=2,
        # is_chat=True,
        is_spectrum=True,
        tokenizer=tokenizer,
    )
    # data = generate_wvs_individual(n_individuals=10_000, seed=2)
    from utils import explore

    explore(data)
    breakpoint()
