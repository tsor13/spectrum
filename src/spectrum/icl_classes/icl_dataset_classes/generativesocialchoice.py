import os
import sys

import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generativesocialchoice_helper_loader(
    file: str, path: str | None = None
) -> pd.DataFrame:
    if path is None:
        path = "data/generativesocialchoice/data/"
    # read in
    # [tsor13@klone-login01 bayesbench]$ ls data/generativesocialchoice/data/
    # abortion_survey.csv  chatbot_personalization_survey.csv
    # files = ['abortion_survey.csv', 'chatbot_personalization_survey.csv']
    # # load data
    # dfs = []
    # for file in files:
    #     dfs.append(pd.read_csv(os.path.join(path, file)))
    # df = pd.concat(dfs)
    df = pd.read_csv(os.path.join(path, file))
    # drop where qeustion_type == reading
    df = df[df["question_type"] != "reading"]
    return df


def generate_generativesocialchoice_validation(
    folder: str | None = None, **kwargs
) -> pd.DataFrame:
    args = {
        "seed": 42,
    }
    args.update(kwargs)
    files = ["abortion_survey.csv", "chatbot_personalization_survey.csv"]
    dfs = []
    for file in files:
        survey_df = generativesocialchoice_helper_loader(file, folder)
        # subset to question_type == "multiple choice + text"
        survey_df = survey_df[survey_df["question_type"] == "multiple choice + text"]
        # do individual multivariate predicting choice, text given question_text and json_choices
        # rename text to response_text
        survey_df = survey_df.rename(columns={"text": "response_text"})
        # strip whitespace for all relevant columns
        survey_df["question_text"] = survey_df["question_text"].str.strip()
        survey_df["response_text"] = survey_df["response_text"].str.strip()
        survey_df["choice"] = survey_df["choice"].str.strip()
        individual_multivariate = IndividualMultivariate(
            survey_df,
            individual_id_column="user_id",
            given_variables=["question_text", "json_choices"],
            gen_variables=["choice", "response_text"],
            name=f"gss-validation/{file}",
            # TODO - could add descriptions
        )
        gens = individual_multivariate.generate_many(**args)
        dfs.append(gens)
    df = pd.concat(dfs)
    return df


def generate_generativesocialchoice_freetext(
    folder: str | None = None, **kwargs
) -> pd.DataFrame:
    args = {
        "seed": 42,
    }
    args.update(kwargs)
    files = ["abortion_survey.csv", "chatbot_personalization_survey.csv"]
    dfs = []
    for file in files:
        survey_df = generativesocialchoice_helper_loader(file, folder)
        # subset to question_type == "text"
        survey_df = survey_df[survey_df["question_type"] == "text"]
        # rename text to response_text
        survey_df = survey_df.rename(columns={"text": "response_text"})
        # strip whitespace for all relevant columns
        survey_df["question_text"] = survey_df["question_text"].str.strip()
        survey_df["response_text"] = survey_df["response_text"].str.strip()
        # drop na for response_text
        survey_df = survey_df.dropna(subset=["response_text"])
        # do individual multivariate predicting response_text given question_text
        individual_multivariate = IndividualMultivariate(
            survey_df,
            individual_id_column="user_id",
            given_variables=["question_text"],
            gen_variables=["response_text"],
            name=f"gss-freetext/{file}",
            # TODO - could add descriptions
        )
        gens = individual_multivariate.generate_many(**args)
        dfs.append(gens)
    df = pd.concat(dfs)
    return df
