import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def load_opinionqa_helper(
    survey_index: int, max_raters_per_survey: int = 400, folder="../data/opinionqa/raw/"
):
    import hashlib

    np.random.seed(0)
    # topic_info = np.load('../data/opinionqa/raw/human_resp/topic_mapping.npy', allow_pickle=True).item()
    topic_info = np.load(
        os.path.join(folder, "human_resp/topic_mapping.npy"), allow_pickle=True
    ).item()
    # add folder/opinions_qa to path
    sys.path.append(os.path.join(folder, "opinions_qa"))
    import helpers as ph
    from helpers import PEW_SURVEY_LIST

    # DATASET_DIR = './data/human_resp/'
    DATASET_DIR = os.path.join(folder, "human_resp/")
    # get all folders in human_resp
    # folders = os.listdir(os.path.join(folder, 'human_resp'))
    # SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST] + \
    # ['Pew_American_Trends_Panel_disagreement_500']
    SURVEY_LIST = [
        f"American_Trends_Panel_W{SURVEY_WAVE}" for SURVEY_WAVE in PEW_SURVEY_LIST
    ]

    if survey_index > len(SURVEY_LIST):
        raise ValueError(
            f"Survey index {survey_index} is out of range for {len(SURVEY_LIST)} surveys."
        )

    instances = []
    rater_metadata = []
    demographic_attributes = ph.DEMOGRAPHIC_ATTRIBUTES[1:]
    alphabet = "abcdefghijklmnopqrstuvwxyz".upper()

    def templatize_question(question, option_mapping):
        return (
            question
            + "\n"
            + "\n".join([f"{resp}: {option_mapping[resp]}" for resp in option_mapping])
        )

    SURVEY_NAME = SURVEY_LIST[survey_index]
    # RESULT_FILES = [f for f in os.listdir(RESULT_DIR) if SURVEY_NAME in f and f'context={CONTEXT}' in f]

    ## Read human responses and survey info
    info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, "info.csv"))
    info_df["option_ordinal"] = info_df.apply(
        lambda x: eval(x["option_ordinal"]), axis=1
    )
    info_df["references"] = info_df.apply(lambda x: eval(x["references"]), axis=1)

    md_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, "metadata.csv"))
    md_df["options"] = md_df.apply(lambda x: eval(x["options"]), axis=1)
    md_order = {"Overall": {"Overall": 0}}
    md_order.update(
        {
            k: {o: oi for oi, o in enumerate(opts)}
            for k, opts in zip(md_df["key"], md_df["options"])
        }
    )

    # ## Get model opinion distribution
    # model_df = ph.get_model_opinions(RESULT_DIR, RESULT_FILES, info_df)
    # read in question in fo from info.csv
    info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, "info.csv"))
    # Index(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'key', 'option_mapping', 'question', 'references', 'option_ordinal'], dtype='object')
    questions = info_df[["key", "question", "option_mapping", "references"]]

    resp_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, "responses.csv"))
    # iterate through resp_df rows and add instances
    # for i, row in resp_df.iterrows():
    # tqdm
    for i, row in tqdm(resp_df.iterrows(), total=len(resp_df)):
        # make a rater_id which is hash of the row string
        # rater_id = hashlib.md5(row.to_string().encode()).hexdigest()
        # loop through all questions
        for question_dict in questions.itertuples():
            # Pandas(Index=0, key='SAFECRIME_W26', question='How safe, if at all, would you say your local community is from crime? Would you say it is', option_mapping="{1.0: 'Very safe', 2.0: 'Somewhat safe', 3.0: 'Not too safe', 4.0: 'Not at all safe', 99.0: 'Refused'}", references="['Very safe', 'Somewhat safe', 'Not too safe', 'Not at all safe', 'Refused']")
            # map references to ABCD etc
            resp_to_alph = dict(zip(eval(question_dict.references), alphabet))
            alph_to_resp = {v: k for k, v in resp_to_alph.items()}
            text_response = row[question_dict.key]
            if text_response not in resp_to_alph:
                continue
            question_text = templatize_question(question_dict.question, alph_to_resp)
            # answer is the response
            answer = resp_to_alph[text_response]
            # add to instances
            # instances.append({'instance_id': SURVEY_NAME + '_' + str(i), 'rater_id': rater_id, 'input': question_text, 'label': answer, 'question': question_dict.question, 'response': text_response, 'num_responses': len(alph_to_resp), 'mapping': alph_to_resp, 'survey': SURVEY_NAME, 'question_key': question_dict.key})
            row_id = SURVEY_NAME + "_" + str(i)
            instances.append(
                {
                    "instance_id": row_id,
                    "rater_id": row_id,
                    "input": question_text,
                    "label": answer,
                    "question": question_dict.question,
                    "response": text_response,
                    "num_responses": len(alph_to_resp),
                    "mapping": alph_to_resp,
                    "survey": SURVEY_NAME,
                    "question_key": question_dict.key,
                }
            )

            # add to rater_metadata
            # rater_info = {'rater_id': rater_id}
            rater_info = {"rater_id": row_id}
            rater_info.update({col: row[col] for col in demographic_attributes})
            rater_metadata.append(rater_info)
    instances = pd.DataFrame(instances)
    # drop duplicates
    # instances = instances.drop_duplicates()
    rater_metadata = pd.DataFrame(rater_metadata)
    # drop duplicates
    rater_metadata = rater_metadata.drop_duplicates()
    # shuffle and keep only the top max_raters_per_survey
    rater_metadata = rater_metadata.sample(frac=1, random_state=0).head(
        max_raters_per_survey
    )
    # only keep the instances with raters in rater_metadata
    instances = instances[instances["rater_id"].isin(rater_metadata["rater_id"])]

    # # make metadata_info (all demographic_attributes are categorical)
    # metadata_info = pd.DataFrame(demographic_attributes, columns=['column_name'])
    # metadata_info['column_type'] = 'categorical'
    # # save and verify
    # task_type = {
    #     "task_type": "classification",
    #     "target_column": "label",
    #     "target_type": "categorical",
    #     "number_of_classes": instances['label'].nunique(),
    #     "class_names": sorted(instances['label'].unique().tolist()),
    #     "input_columns": ["input"],
    #     "instruction_text": "Answer the survey question from the perspective of the following rater."
    # }
    return {
        "instances": instances,
        "rater_metadata": rater_metadata,
    }


# def load_opinionqa_question_helper(folder: str | None = None, survey_index: int = 0) -> pd.DataFrame:
def load_opinionqa_question_helper(
    survey_index: int, folder: str | None = None
) -> pd.DataFrame:
    """
    Load the opinionqa question helper.
    """
    import hashlib

    np.random.seed(0)
    # topic_info = np.load('../data/opinionqa/raw/human_resp/topic_mapping.npy', allow_pickle=True).item()
    topic_info = np.load(
        os.path.join(folder, "human_resp/topic_mapping.npy"), allow_pickle=True
    ).item()
    # add folder/opinions_qa to path
    sys.path.append(os.path.join(folder, "opinions_qa"))
    import helpers as ph
    from helpers import PEW_SURVEY_LIST

    # DATASET_DIR = './data/human_resp/'
    DATASET_DIR = os.path.join(folder, "human_resp/")
    # get all folders in human_resp
    # folders = os.listdir(os.path.join(folder, 'human_resp'))
    # SURVEY_LIST = [f'American_Trends_Panel_W{SURVEY_WAVE}' for SURVEY_WAVE in PEW_SURVEY_LIST] + \
    # ['Pew_American_Trends_Panel_disagreement_500']
    SURVEY_LIST = [
        f"American_Trends_Panel_W{SURVEY_WAVE}" for SURVEY_WAVE in PEW_SURVEY_LIST
    ]

    if survey_index > len(SURVEY_LIST):
        raise ValueError(
            f"Survey index {survey_index} is out of range for {len(SURVEY_LIST)} surveys."
        )

    instances = []
    rater_metadata = []
    demographic_attributes = ph.DEMOGRAPHIC_ATTRIBUTES[1:]
    alphabet = "abcdefghijklmnopqrstuvwxyz".upper()

    def templatize_question(question, option_mapping):
        return (
            question
            + "\n"
            + "\n".join([f"{resp}: {option_mapping[resp]}" for resp in option_mapping])
        )

    SURVEY_NAME = SURVEY_LIST[survey_index]

    ## Read human responses and survey info
    info_df = pd.read_csv(os.path.join(DATASET_DIR, SURVEY_NAME, "info.csv"))
    info_df["option_ordinal"] = info_df.apply(
        lambda x: eval(x["option_ordinal"]), axis=1
    )
    info_df["references"] = info_df.apply(lambda x: eval(x["references"]), axis=1)
    # info_df['input'] = info_df.apply(lambda x: templatize_question(x['question'], x['option_mapping']), axis=1)
    alphabet = "abcdefghijklmnopqrstuvwxyz".upper()
    info_df["resp_to_alph"] = info_df.apply(
        lambda x: dict(zip(x["references"], alphabet)), axis=1
    )
    info_df["alph_to_resp"] = info_df.apply(
        lambda x: {v: k for k, v in x["resp_to_alph"].items()}, axis=1
    )
    info_df["input"] = info_df.apply(
        lambda x: templatize_question(x["question"], x["alph_to_resp"]), axis=1
    )
    return info_df


def generate_opinionqa_data(
    n_surveys: int = 15,
    max_raters_per_survey: int = 200,
    folder: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate the opinionqa data.
    """
    args = {
        "seed": 42,
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/opinionqa/"
    # load the data
    dfs = []
    for survey_ind in tqdm(range(n_surveys)):
        # oqa_data = load_opinionqa_helper(0, max_raters_per_survey, folder)
        oqa_data = load_opinionqa_helper(survey_ind, max_raters_per_survey, folder)
        instances = oqa_data["instances"]
        # rename input to question
        instances.rename(columns={"input": "Question"}, inplace=True)
        # rename label to response
        instances.rename(columns={"label": "Response"}, inplace=True)
        rater_metadata = oqa_data["rater_metadata"]
        # set numpy seed
        np.random.seed(args["seed"])
        # for rater_id in rater_metadata['rater_id']:
        for i, rater_id in enumerate(rater_metadata["rater_id"]):
            # get the instances for the rater
            rater_instances = instances[instances["rater_id"] == rater_id]
            # shuffle the instances
            # rater_instances = rater_instances.sample(frac=1, random_state=args['seed']+i).reset_index(drop=True)
            # get the metadata
            metadata = rater_metadata[rater_metadata["rater_id"] == rater_id].to_dict(
                orient="records"
            )[0]
            # remove rater_id from metadata
            metadata.pop("rater_id")
            # for each key in metadata, pop with p 0.3
            for key in list(metadata.keys()):
                if np.random.rand() < 0.3:
                    metadata.pop(key)
            description = [
                "The following are survey responses for a single individual.",
                "Survey responses from the following individual: " + str(metadata),
            ]
            # rename input to question
            # rater_instances.rename(columns={'input': 'Question'}, inplace=True)
            # rater_instances['Question'] = rater_instances['input']
            # # rename label to response
            # rater_instances['Response'] = rater_instances['label']
            # do generic multivariate
            # TODO - if I were wanting to spend more time on this, I might try to
            rater_multivariate = GenericMultivariate(
                df=rater_instances,
                given_variables=["Question"],
                gen_variables=["Response"],
                descriptions=description,
                name=f"opinionqa_rater_{rater_id}",
                do_shuffle=False,
            )
            args["seed"] = args["seed"] + 1
            rater_gens = rater_multivariate.generate_many(**args)
            dfs.append(rater_gens)
    data = pd.concat(dfs)
    # shuffle
    data = data.sample(frac=1, random_state=args["seed"]).reset_index(drop=True)
    return data


def generate_opinionqa_question_data(
    folder: str | None = None, **kwargs
) -> pd.DataFrame:
    """
    Generate the opinionqa question data.
    """
    args = {
        "seed": 42,
    }
    args.update(kwargs)
    if folder is None:
        folder = "data/opinionqa/"
    dfs = []
    for survey_ind in range(15):
        question_data = load_opinionqa_question_helper(survey_ind, folder)
        questions = question_data["input"].tolist()
        # to categorical
        categorical = SingleVariableIID(
            questions,
            name=f"opinionqa_question_{survey_ind}",
            descriptions=["The following are survey questions."],
            replacement=False,
        )
        args["seed"] = args["seed"] + 1
        question_gens = categorical.generate_many(**args)
        dfs.append(question_gens)
    data = pd.concat(dfs)
    return data
