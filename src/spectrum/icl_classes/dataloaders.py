"""
Quick checks while iterating on loaders:

uv run src/spectrum/icl_classes/dataloaders.py --dataset dices --format spectrum --tokenizer qwen/Qwen3-14B
uv run src/spectrum/icl_classes/dataloaders.py --dataset habermas_individual --format spectrum --tokenizer qwen/Qwen3-14B
uv run src/spectrum/icl_classes/dataloaders.py --dataset coinflip --format spectrum --tokenizer google/gemma-3-1b-it
uv run src/spectrum/icl_classes/dataloaders.py --dataset dices --format colon --tokenizer google/gemma-3-1b-it
"""

import html
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from spectrum.format_utils import TEMPLATE_MAPS
from spectrum.icl_classes.icl_dataset_classes.ambient import (
    generate_ambient_ambiguity_detection,
    generate_ambient_annotation_distributions,
    generate_ambient_disambiguation,
    generate_ambient_interpretation_labels,
    generate_ambient_linguist_annotations,
    generate_ambient_premise_hypothesis,
)
from spectrum.icl_classes.icl_dataset_classes.arc import generate_arc
from spectrum.icl_classes.icl_dataset_classes.babynames import generate_babynames
from spectrum.icl_classes.icl_dataset_classes.bare import (
    generate_bare_enron,
    generate_bare_gsm8k,
    generate_bare_hotpot,
    generate_bare_lcb,
    generate_bare_newsgroups,
    generate_bare_pubmed,
)
from spectrum.icl_classes.icl_dataset_classes.changemyview import (
    generate_cmv_categories,
    generate_cmv_comments,
    generate_cmv_posts,
)
from spectrum.icl_classes.icl_dataset_classes.chatbotarena import (
    generate_chatbotarena_assistant,
    generate_chatbotarena_individual_prefs,
    generate_chatbotarena_prompts,
)
from spectrum.icl_classes.icl_dataset_classes.chemistry import (
    generate_chemistry_esol,
    generate_chemistry_oxidative,
)
from spectrum.icl_classes.icl_dataset_classes.coinflip import generate_coinflip_data
from spectrum.icl_classes.icl_dataset_classes.collective_alignment import (
    generate_collective_alignment_individual,
)
from spectrum.icl_classes.icl_dataset_classes.community_alignment import (
    generate_community_alignment_individual_preferences,
    generate_community_alignment_individual_reply,
    generate_community_alignment_initial_prompt,
    generate_community_alignment_response,
)
from spectrum.icl_classes.icl_dataset_classes.dices import generate_dices
from spectrum.icl_classes.icl_dataset_classes.diffuse_distributions import (
    generate_diffuse_distributions_data,
)
from spectrum.icl_classes.icl_dataset_classes.discrete_dists import (
    generate_binomial,
    generate_cards,
    generate_categorical,
    generate_geometric,
    generate_geometric_beta,
    generate_hypergeometric,
    generate_multinomial,
    generate_negative_binomial,
    generate_poisson,
    generate_zipfian,
)
from spectrum.icl_classes.icl_dataset_classes.drop import generate_drop
from spectrum.icl_classes.icl_dataset_classes.generativesocialchoice import (
    generate_generativesocialchoice_freetext,
    generate_generativesocialchoice_validation,
)
from spectrum.icl_classes.icl_dataset_classes.globaloqa import generate_globaloqa_data
from spectrum.icl_classes.icl_dataset_classes.gpqa import generate_gpqa
from spectrum.icl_classes.icl_dataset_classes.gsm8k import (
    generate_gsm8k_answer_from_question,
    generate_gsm8k_question,
    generate_gsm8k_question_answer,
    generate_gsm8k_question_from_answer,
)
from spectrum.icl_classes.icl_dataset_classes.habermas import (
    generate_habermas_individual_categorical_data,
    generate_habermas_individual_data,
    generate_habermas_opinions_data,
    generate_habermas_question_data,
)
from spectrum.icl_classes.icl_dataset_classes.haikus import generate_haikus
from spectrum.icl_classes.icl_dataset_classes.hatespeech import (
    generate_hatespeech_individual_data,
)
from spectrum.icl_classes.icl_dataset_classes.hellaswag import generate_hellaswag
from spectrum.icl_classes.icl_dataset_classes.helpsteer import generate_helpsteer
from spectrum.icl_classes.icl_dataset_classes.imdb import (
    generate_imdb,
    generate_imdb_individual,
)
from spectrum.icl_classes.icl_dataset_classes.issuebench import generate_issuebench
from spectrum.icl_classes.icl_dataset_classes.jeopardy import (
    generate_jeopardy_answer_prediction,
    generate_jeopardy_question_generation,
)
from spectrum.icl_classes.icl_dataset_classes.lewidi_csc import (
    generate_csc_sarcasm_detection_individual,
)
from spectrum.icl_classes.icl_dataset_classes.lewidi_mp import (
    generate_mp_irony_detection_individual,
)
from spectrum.icl_classes.icl_dataset_classes.lewidi_par import (
    generate_par_paraphrase_detection_individual,
    generate_par_paraphrase_detection_individual_categorical,
)
from spectrum.icl_classes.icl_dataset_classes.lewidi_varierrnli import (
    generate_varierrnli_nli_detection_individual,
    generate_varierrnli_nli_detection_individual_categorical,
)
from spectrum.icl_classes.icl_dataset_classes.mmlu import generate_mmlu
from spectrum.icl_classes.icl_dataset_classes.netflix import (
    generate_netflix_individual_ratings,
    generate_netflix_individual_views,
)
from spectrum.icl_classes.icl_dataset_classes.normal import generate_normal_data
from spectrum.icl_classes.icl_dataset_classes.novacomet import (
    generate_novacomet_hypothesis_data,
    generate_novacomet_premise_data,
)
from spectrum.icl_classes.icl_dataset_classes.numbergame import (
    generate_numbergame_individual,
    generate_numbergame_perc,
)
from spectrum.icl_classes.icl_dataset_classes.opinionqa import (
    generate_opinionqa_data,
    generate_opinionqa_question_data,
)
from spectrum.icl_classes.icl_dataset_classes.polis import (
    generate_polis_comment_data,
    generate_polis_vote_data,
)
from spectrum.icl_classes.icl_dataset_classes.popquorn import (
    generate_popquorn_individual,
    generate_popquorn_og_categorical,
)
from spectrum.icl_classes.icl_dataset_classes.prism import (
    generate_prism_individual_preferences,
    generate_prism_opening_prompts_individual,
    generate_prism_prompts,
)
from spectrum.icl_classes.icl_dataset_classes.synth_flight_prefs import (
    generate_synth_flight_prefs,
)
from spectrum.icl_classes.icl_dataset_classes.titanic import (
    generate_titanic_all_variables,
    generate_titanic_survival_prediction,
)
from spectrum.icl_classes.icl_dataset_classes.truthful_qa import generate_truthful_qa_mc
from spectrum.icl_classes.icl_dataset_classes.valueconsistency import generate_vc
from spectrum.icl_classes.icl_dataset_classes.valueprism import (
    generate_valueprism_misc,
    generate_valueprism_situations,
    generate_valueprism_vrd_data,
    generate_valueprism_vrds_noncontextual,
)
from spectrum.icl_classes.icl_dataset_classes.winogrande import generate_winogrande
from spectrum.icl_classes.icl_dataset_classes.wvs import generate_wvs_individual

default_data_map = {
    "polis_vote": generate_polis_vote_data,
    "polis_comment": generate_polis_comment_data,
    "diffuse_distribution": generate_diffuse_distributions_data,
    "coinflip": generate_coinflip_data,
    "normal": generate_normal_data,
    "novacomet_hypothesis": generate_novacomet_hypothesis_data,
    "novacomet_premise": generate_novacomet_premise_data,
    "habermas_question": generate_habermas_question_data,
    "habermas_individual": generate_habermas_individual_data,
    "habermas_individual_categorical": generate_habermas_individual_categorical_data,
    "hatespeech_individual": generate_hatespeech_individual_data,
    "valueprism_situation": generate_valueprism_situations,
    "valueprism_vrd": generate_valueprism_vrd_data,
    "valueprism_vrds_noncontextual": generate_valueprism_vrds_noncontextual,
    "valueprism_misc": generate_valueprism_misc,
    "opinionqa_individual": generate_opinionqa_data,
    "opinionqa_questions": generate_opinionqa_question_data,
    "wvs_individual": generate_wvs_individual,
    "generativesocialchoice_validation": generate_generativesocialchoice_validation,
    "generativesocialchoice_freetext": generate_generativesocialchoice_freetext,
    "popquorn_individual": generate_popquorn_individual,
    "popquorn_og_categorical": generate_popquorn_og_categorical,
    "numbergame_individual": generate_numbergame_individual,
    "numbergame_perc": generate_numbergame_perc,
    "babynames": generate_babynames,
    "haikus": generate_haikus,
    "valueconsistency": generate_vc,
    "globaloqa": generate_globaloqa_data,
    "issuebench": generate_issuebench,
    "chatbotarena_prompts": generate_chatbotarena_prompts,
    "chatbotarena_individual_prefs": generate_chatbotarena_individual_prefs,
    "chatbotarena_assistant": generate_chatbotarena_assistant,
    "helpsteer": generate_helpsteer,
    "dices": generate_dices,
    "cards": generate_cards,
    "multinomial": generate_multinomial,
    "geometric": generate_geometric,
    "poisson": generate_poisson,
    "binomial": generate_binomial,
    "negative_binomial": generate_negative_binomial,
    "categorical": generate_categorical,
    "geometric_beta": generate_geometric_beta,
    "hypergeometric": generate_hypergeometric,
    "zipfian": generate_zipfian,
    "imdb": generate_imdb,
    # "imdb_individual": generate_imdb_individual,
    "flight": generate_synth_flight_prefs,
    "changemyview_categories": generate_cmv_categories,
    "changemyview_posts": generate_cmv_posts,
    # From BARE
    "bare_gsm8k": generate_bare_gsm8k,
    "bare_enron": generate_bare_enron,
    "bare_hotpot": generate_bare_hotpot,
    "bare_lcb": generate_bare_lcb,
    "newsgroups": generate_bare_newsgroups,
    "pubmed": generate_bare_pubmed,
    # Real GSM8K dataset
    "gsm8k_question": generate_gsm8k_question,
    "gsm8k_answer_from_question": generate_gsm8k_answer_from_question,
    "gsm8k_question_answer": generate_gsm8k_question_answer,
    "gsm8k_question_from_answer": generate_gsm8k_question_from_answer,
    # AmbiEnt dataset
    "ambient_premise_hypothesis": generate_ambient_premise_hypothesis,
    "ambient_ambiguity_detection": generate_ambient_ambiguity_detection,
    "ambient_disambiguation": generate_ambient_disambiguation,
    "ambient_linguist_annotations": generate_ambient_linguist_annotations,
    "ambient_interpretation_labels": generate_ambient_interpretation_labels,
    "ambient_annotation_distributions": generate_ambient_annotation_distributions,
    # Titanic dataset
    "titanic_all_variables": generate_titanic_all_variables,
    "titanic_survival_prediction": generate_titanic_survival_prediction,
    # Jeopardy dataset
    "jeopardy_question_generation": generate_jeopardy_question_generation,
    "jeopardy_answer_prediction": generate_jeopardy_answer_prediction,
    # PRISM dataset
    "prism_prompts": generate_prism_prompts,
    "prism_prompts_individual": generate_prism_opening_prompts_individual,
    "prism_individual_preferences": generate_prism_individual_preferences,
    # Netflix dataset
    "netflix_individual_ratings": generate_netflix_individual_ratings,
    "netflix_individual_views": generate_netflix_individual_views,
    "lewidi_csc_sarcasm_detection_individual": generate_csc_sarcasm_detection_individual,
    "lewidi_mp_irony_detection_individual": generate_mp_irony_detection_individual,
    "lewidi_par_paraphrase_detection_individual": generate_par_paraphrase_detection_individual,
    "lewidi_par_paraphrase_detection_individual_categorical": generate_par_paraphrase_detection_individual_categorical,
    "lewidi_par_paraphrase_detection_individual": generate_par_paraphrase_detection_individual,
    "lewidi_varierrnli_nli_detection_individual": generate_varierrnli_nli_detection_individual,
    "lewidi_varierrnli_nli_detection_individual_categorical": generate_varierrnli_nli_detection_individual_categorical,
    # Community Alignment dataset
    "community_alignment_initial_prompt": generate_community_alignment_initial_prompt,
    "community_alignment_individual_reply": generate_community_alignment_individual_reply,
    "community_alignment_individual_preferences": generate_community_alignment_individual_preferences,
    "community_alignment_response": generate_community_alignment_response,
    # Collective Alignment dataset
    "collective_alignment_individual": generate_collective_alignment_individual,
    # MMLU dataset
    "mmlu": generate_mmlu,
    # GPQA dataset
    "gpqa": generate_gpqa,
    # HellaSwag dataset
    "hellaswag": generate_hellaswag,
    # Winogrande dataset
    "winogrande": generate_winogrande,
    # ARC dataset
    "arc": generate_arc,
    # DROP dataset
    "drop": generate_drop,
    # TruthfulQA dataset
    "truthful_qa": generate_truthful_qa_mc,
    # Chemistry dataset
    "chemistry_esol": generate_chemistry_esol,
    "chemistry_oxidative": generate_chemistry_oxidative,
}

include_in_test = [
    "wvs",
    "globaloqa",
    "wvs",
    "habermas",
    "numbergame",
    "flight",
    "chatbotarena",
    "novacomet",
    "mmlu",
    "hellaswag",
    "winogrande",
    "arc",
    "drop",
    "truthful_qa",
    "gpqa",
    "chemistry_esol",
    "chemistry_oxidative",
]

all_names = list(default_data_map.keys())


def match_to_include_in_test(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix for prefix in include_in_test)


val_defaults = [name for name in all_names if match_to_include_in_test(name)]
# print("Val defaults:")
# print(val_defaults)
train_defaults = [name for name in all_names if name not in val_defaults]
# print("Train defaults:")
# print(train_defaults)


def load_default_data(name: str, shuffle: bool = True, **kwargs) -> pd.DataFrame:
    data = default_data_map[name](**kwargs)
    seed = kwargs.get("seed", 42)
    if shuffle:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    return data


def debug(name: str, format: str, tokenizer_name: str = "google/gemma-3-1b-it"):
    from spectrum.format_utils import show_loss_texts

    # load the data
    data = load_default_data(name, include_description_prob=0.5)
    # print(data)
    template = TEMPLATE_MAPS[format]["messages_to_loss_texts"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = template(messages=data.iloc[0]["messages"][:11], tokenizer=tokenizer)
    print(show_loss_texts(texts))
    print("Length of data:")
    print(len(data))
    breakpoint()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="google/gemma-3-1b-it")
    args = parser.parse_args()
    dataset = args.dataset
    format = args.format
    tokenizer_name = args.tokenizer
    debug(dataset, format, tokenizer_name)
