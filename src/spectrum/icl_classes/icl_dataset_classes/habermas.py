"""
uv run src/spectrum/icl_classes/icl_dataset_classes/habermas.py
"""

import json
import os

import pandas as pd

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def habermas_helper_loader(path="raw_data/habermas_data/"):
    # if 'habermas' in dataset_cache:
    #     return dataset_cache['habermas'].copy()
    # position statement ratings
    ratings_path = os.path.join(path, "hm_all_position_statement_ratings.parquet")
    df = pd.read_parquet(ratings_path)
    # get ratings.agreement
    # change ratings agreement to string
    df["ratings.agreement"] = df["ratings.agreement"].astype(str)
    # ['AGREE']                8523
    # ['STRONGLY_AGREE']       8068
    # ['MOCK']                 5805
    # ['STRONGLY_DISAGREE']    5773
    # ['SOMEWHAT_AGREE']       5641
    # ['DISAGREE']             5174
    # ['NEUTRAL']              3717
    # ['SOMEWHAT_DISAGREE']    3275
    # remove [' and ']
    df["ratings.agreement"] = (
        df["ratings.agreement"].str.replace("['", "").str.replace("']", "")
    )
    # drop mock
    df = df[df["ratings.agreement"] != "MOCK"]
    # map onto Agree, Strongly Agree, Disagree, Strongly Disagree...
    df["ratings.agreement"] = df["ratings.agreement"].map(
        {
            "AGREE": "Agree",
            "STRONGLY_AGREE": "Strongly Agree",
            "DISAGREE": "Disagree",
            "STRONGLY_DISAGREE": "Strongly Disagree",
            "SOMEWHAT_AGREE": "Somewhat Agree",
            "SOMEWHAT_DISAGREE": "Somewhat Disagree",
            "NEUTRAL": "Neutral",
        }
    )
    # Index(['index', 'monotonic_timestamp', 'launch_id', 'metadata.created',
    #    'metadata.id', 'metadata.participant_id', 'metadata.provenance',
    #    'metadata.response_duration', 'metadata.status',
    #    'metadata.task_duration', 'metadata.updated', 'metadata.version',
    #    'question.affirming_statement', 'question.id',
    #    'question.negating_statement', 'question.split', 'question.text',
    #    'question.topic', 'question_index', 'rating_index', 'ratings.agreement',
    #    'ratings.metadata.created', 'ratings.metadata.id',
    #    'ratings.metadata.participant_id', 'timestamp', 'worker_id'],
    #   dtype='object')
    # keep only question.text, question.affirming_statement, question.negating_statement, ratings.agreement, question.id, worker_id
    df = df[
        [
            "question.text",
            "question.affirming_statement",
            "question.negating_statement",
            "ratings.agreement",
            "question.id",
            "worker_id",
        ]
    ]
    # drop where worker_id is None
    df = df[df["worker_id"].notna()]
    # rename columns
    # df.columns = ['question', 'affirming_statement', 'negating_statement', 'agreement', 'question_id', 'worker_id']

    # read in statement ratings
    survey_path = os.path.join(path, "hm_all_round_survey_responses.parquet")
    df_survey = pd.read_parquet(survey_path)
    df_survey = df_survey[["worker_id", "question.id", "opinion.text"]]
    # merge on worker_id and question id
    df = df.merge(df_survey, on=["worker_id", "question.id"], how="left")
    # deduplicate on worker_id and question.id
    df = df.drop_duplicates(subset=["worker_id", "question.id"])
    # drop na
    df = df.dropna()
    # dataset_cache['habermas'] = df
    return df


hab_possible_ratings = [
    "Strongly Agree",
    "Agree",
    "Somewhat Agree",
    "Neutral",
    "Somewhat Disagree",
    "Disagree",
    "Strongly Disagree",
]


def load_habermas_raw(path="raw_data/habermas_data/", kind="text"):
    raw_df = habermas_helper_loader(path)
    possible_ratings = raw_df["ratings.agreement"].unique()
    # assert that possible_ratings is a subset of hab_possible_ratings
    assert set(possible_ratings).issubset(set(hab_possible_ratings))

    def get_pm(ratings):
        d = {p: 0 for p in possible_ratings}
        for r in ratings:
            if r not in d:
                d[r] = 0
            d[r] += 1
        return d

    if kind == "text":
        # group by question.text and get list of opinion.text
        df = raw_df.groupby("question.text")["opinion.text"].apply(list).reset_index()
        df["n_responses"] = df["opinion.text"].apply(len)
        df["input"] = df["question.text"]
        df["samples"] = df["opinion.text"]
        df["group"] = "UK residents"
        return df.reset_index(drop=True)
    elif kind == "categorical":
        df = (
            raw_df.groupby("question.text")["ratings.agreement"]
            .apply(list)
            .reset_index()
        )
        df["input"] = df["question.text"]
        # get pmf
        df["pmf"] = df["ratings.agreement"].apply(get_pm)
        df["options"] = [possible_ratings] * len(df)
        df["group"] = "UK residents"
        return df.reset_index(drop=True)
    elif kind == "text_categorical":
        # make a combined column with question.text and ratings.agreement
        raw_df["text_categorical"] = (
            raw_df["opinion.text"]
            + "\nCategorical response: "
            + raw_df["ratings.agreement"]
        )
        # df = raw_df.groupby('question.text')[['text_categorical', 'ratings.agreement']].apply(list).reset_index()
        df = (
            raw_df[["question.text", "text_categorical", "ratings.agreement"]]
            .groupby("question.text")
            .agg({"text_categorical": list, "ratings.agreement": list})
            .reset_index()
        )
        df["input"] = df["question.text"]
        df["samples"] = df["text_categorical"]
        df["pmf"] = df["ratings.agreement"].apply(get_pm)
        df["n_responses"] = df["text_categorical"].apply(len)
        df["options"] = [possible_ratings] * len(df)
        df["group"] = "UK residents"
        return df.reset_index(drop=True)
    else:
        raise ValueError(f"Invalid kind: {kind}")


def generate_habermas_question_data(
    folder: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Predict the questions.
    """
    args = {
        "seed": 42,
        # "n_per": 100,
        # "n_iter": geom_and_poisson_iter(mean=128),
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/habermas_data"
    # read in all json files
    habermas_df = load_habermas_raw(path=folder, kind="text")
    questions = habermas_df["question.text"].unique().tolist()
    # make categorical
    categorical = SingleVariableIID(
        questions,
        name="habermas_question",
        descriptions=[
            "Generate a question.",
            "Questions",
            "Generate a list of diverse questions.",
            "Generate questions that could be posed to UK residents.",
            "Questions about UK politics",
        ],
        replacement=False,
    )
    data = categorical.generate_many(**args)
    return data


def generate_habermas_opinions_data(
    folder: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Given a question, predict multiple opinions.
    """
    args = {
        "seed": 42,
        # "n_per": 200,
        # "n_iter": geom_and_poisson_iter(mean=16),
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=8),
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/habermas_data"
    habermas_df = load_habermas_raw(path=folder, kind="text")
    # group by question.text and get list of opinion.text
    df = habermas_df.groupby("question.text")["opinion.text"].apply(list).reset_index()
    df["question"] = df["question.text"]
    df["opinions"] = df["opinion.text"]
    df = df[["question", "opinions"]]
    # max opinions per question
    max_opinions = 10
    df["opinions"] = df["opinions"].apply(lambda x: x[0][:max_opinions])
    # dropna
    df = df.dropna()
    # drop where no opinions
    df = df[df["opinions"].apply(len) > 0]
    # to string
    # df['opinions2'] = df['opinions'].apply(lambda x: "; ".join(x[0]))
    # to string
    df["opinions"] = df["opinions"].apply(str)
    # make generic multivariate
    generic_multivariate = GenericMultivariate(
        df=df,
        given_variables=["question"],
        gen_variables=["opinions"],
        name="habermas_opinions",
        descriptions=[
            "The following are opinions of UK residents. (list of 2-3 sentence opinions)",
            "Opinions from representative sample of UK residents.",
            "Opinions",
        ],
    )
    data = generic_multivariate.generate_many(**args)
    # TODO - it seems like it's grouping opinion and question together?
    return data


def generate_habermas_individual_data(
    folder: str | None = None,
    # seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Predict an individual's ratings, both categorical and text.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 2000,
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/habermas_data"
    habermas_df = habermas_helper_loader(path=folder)
    # shuffle order
    habermas_df = habermas_df.sample(frac=1, random_state=args["seed"] + 2).reset_index(
        drop=True
    )
    # rename question.affirming_statement to statement, opinion.text to opinion
    habermas_df.rename(
        columns={"question.affirming_statement": "statement"}, inplace=True
    )
    # make individual multivariate
    individual_multivariate = IndividualMultivariate(
        df=habermas_df,
        individual_id_column="worker_id",
        # individual_description_column='ratings.agreement',
        given_variables=["question.text", "statement"],
        # gen_variables=['opinion.text', 'ratings.agreement'],
        gen_variables=["ratings.agreement", "opinion.text"],
        descriptions=[
            "UK resident responses. They were given a question and a statement, asked to express their opinion in 2-3 sentences (opinion.text) and their level of agreement with it on a 7-point scale (ratings.agreement).",
            "UK resident responses.\nGiven:\n-question.text: A question to respond to.\n-statement: A statement to respond to.\nOutputs:\n-opinion.text: 2-3 sentences\n-ratings.agreement: Strongly Agree; Agree; Somewhat Agree; Neutral; Somewhat Disagree; Disagree; Strongly Disagree",
        ],
        name="habermas_individual",
    )
    data = individual_multivariate.generate_many(**args)
    return data


def generate_habermas_individual_categorical_data(
    folder: str | None = None,
    # seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """
    Predict an individual's categorical ratings.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 2000,
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/habermas_data"
    # add 1 to the seed to avoid the same seed for different tasks
    args["seed"] += 1
    habermas_df = habermas_helper_loader(path=folder)
    # shuffle order
    habermas_df = habermas_df.sample(frac=1, random_state=args["seed"] + 1).reset_index(
        drop=True
    )
    # rename question.affirming_statement to statement, opinion.text to opinion
    habermas_df.rename(
        columns={"question.affirming_statement": "statement"}, inplace=True
    )
    # make individual multivariate
    individual_multivariate = IndividualMultivariate(
        df=habermas_df,
        individual_id_column="worker_id",
        # individual_description_column='ratings.agreement',
        given_variables=["question.text", "statement"],
        gen_variables=["ratings.agreement"],
        descriptions=[
            "Given a question and a statement, predict the level of agreement with it on a 7-point scale.\nOptions: Strongly Agree; Agree; Somewhat Agree; Neutral; Somewhat Disagree; Disagree; Strongly Disagree"
        ],
        name="habermas_individual_categorical",
    )
    data = individual_multivariate.generate_many(**args)
    return data
