import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter


def load_gsm8k_data(file_path: str | None = None):
    """Load GSM8K synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/gsm8k/gsm8k_gpt_4o_refine_llama_70b_base_n=1000.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def load_enron_data(file_path: str | None = None):
    """Load Enron synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/enron/enron_gpt_4o_refine_llama_70b_base_n=500.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            # Flatten the nested email structure
            flattened_item = {"spam": item["spam"], "email": item["email"]}
            data.append(flattened_item)

    return pd.DataFrame(data)


def generate_bare_gsm8k(mode: str = "all", **kwargs):
    """Generate GSM8K dataset with specified mode.

    Args:
        mode: One of "all", "question_answer", "answer_from_question", "question_from_answer", "question"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 42,
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_inds": 50,  # max of 50 per (they're quite long)
        "max_total": 500,
    }
    args.update(kwargs)

    df = load_gsm8k_data()
    dfs = []

    if mode in ["all", "question_answer"]:
        # Predict both question and answer (no given variables)
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["question", "answer"],
            name="gsm8k_question_answer",
            descriptions=[
                "Generate both a math word problem (question) and its step-by-step solution (answer). These are synthetic grade school math problems covering arithmetic, basic algebra, and word problem reasoning.",
                "Synthetic math word problems with worked solutions. Generate both the question and answer components separately.",
                "Create complete math problem pairs: generate both the word problem statement and its detailed solution with calculations.",
            ],
            seed=args["seed"] + 1,
        )
        question_answer_args = args.copy()
        question_answer_args["max_inds"] = 50  # Override for this mode
        dfs.append(generator.generate_many(**question_answer_args))

    if mode in ["all", "answer_from_question"]:
        # Predict answer given question
        generator = GenericMultivariate(
            df=df,
            given_variables=["question"],
            gen_variables=["answer"],
            name="gsm8k_answer_from_question",
            descriptions=[
                "Given a math word problem, generate its step-by-step solution with calculations and reasoning.",
                "Solve grade school math word problems by providing detailed worked solutions.",
                "Given: question (math word problem). Generate: answer (step-by-step solution with final numerical answer).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question_from_answer"]:
        # Predict question given answer
        generator = GenericMultivariate(
            df=df,
            given_variables=["answer"],
            gen_variables=["question"],
            name="gsm8k_question_from_answer",
            descriptions=[
                "Given a step-by-step math solution, generate a word problem that would lead to this solution.",
                "Create math word problems that match the given worked solution and final answer.",
                "Given: answer (step-by-step solution). Generate: question (word problem that leads to this solution).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question"]:
        # Predict just the question (no given variables)
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["question"],
            name="gsm8k_question",
            descriptions=[
                "Generate math word problems suitable for grade school students, covering arithmetic and basic algebra.",
                "Create diverse math word problems involving real-world scenarios and numerical reasoning.",
                "Generate: question (math word problem statement only, without solution).",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'question_answer', 'answer_from_question', 'question_from_answer', 'question'"
        )

    return pd.concat(dfs, ignore_index=True)


def generate_bare_enron(mode: str = "all", **kwargs):
    """Generate Enron email classification dataset with specified mode.

    Args:
        mode: One of "all", "spam_email", "spam_from_email", "email_from_spam", "email"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 100,  # Different base seed from GSM8K
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_total": 500,
        "max_inds": 50,  # Smaller since only 500 total examples
    }
    args.update(kwargs)

    df = load_enron_data()
    # change spam to string
    df["spam"] = df["spam"].astype(str)
    dfs = []

    if mode in ["all", "spam_email"]:
        # Predict both spam classification and email content
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["spam", "email"],
            name="enron_spam_email",
            descriptions=[
                "Generate both an email message and its spam classification (true/false). These are synthetic emails based on the Enron dataset.",
                "Create complete email examples: generate both the email content and whether it should be classified as spam.",
                "Generate: spam (boolean), email (message content). Synthetic email classification dataset.",
            ],
            seed=args["seed"] + 1,
        )
        spam_email_args = args.copy()
        spam_email_args["max_inds"] = 50  # Override for this mode
        dfs.append(generator.generate_many(**spam_email_args))

    if mode in ["all", "spam_from_email"]:
        # Predict spam classification given email content
        generator = GenericMultivariate(
            df=df,
            given_variables=["email"],
            gen_variables=["spam"],
            name="enron_spam_from_email",
            descriptions=[
                "Given an email message, classify whether it is spam (true) or not spam (false).",
                "Email spam classification: determine if the given email content should be marked as spam.",
                "Given: email (message content). Generate: spam (boolean classification).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "email_from_spam"]:
        # Predict email content given spam classification
        generator = GenericMultivariate(
            df=df,
            given_variables=["spam"],
            gen_variables=["email"],
            name="enron_email_from_spam",
            descriptions=[
                "Given a spam classification (true/false), generate an email message that matches that classification.",
                "Create email content that corresponds to the specified spam/not-spam label.",
                "Given: spam (boolean). Generate: email (message content that matches the spam classification).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "email"]:
        # Predict just the email content (no given variables)
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["email"],
            name="enron_email",
            descriptions=[
                "Generate email messages in the style of business/corporate communications from the Enron dataset.",
                "Create realistic email content covering business communications, meeting notes, and corporate correspondence.",
                "Generate: email (message content only). Synthetic emails based on business communications.",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'spam_email', 'spam_from_email', 'email_from_spam', 'email'"
        )

    return pd.concat(dfs, ignore_index=True)


def load_hotpot_data(file_path: str | None = None):
    """Load HotpotQA synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/hotpot/hotpot_gpt_4o_refine_llama_70b_base_n=1000.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def generate_bare_hotpot(mode: str = "all", **kwargs):
    """Generate HotpotQA dataset with specified mode.

    Args:
        mode: One of "all", "context_question_answer", "answer_from_context_question", "question_from_context_answer", "context"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 200,  # Different base seed
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_total": 500,
        "max_inds": 50,
    }
    args.update(kwargs)

    df = load_hotpot_data()
    dfs = []

    if mode in ["all", "context_question_answer"]:
        # Predict context, question, and answer
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["context", "question", "answer"],
            name="hotpot_context_question_answer",
            descriptions=[
                "Generate complete reading comprehension examples: context passage, question, and answer. These are synthetic multi-hop reasoning questions requiring information from multiple sources.",
                "Create HotpotQA-style reading comprehension: generate context (background information), question (multi-hop reasoning), and answer (with supporting evidence).",
                "Generate: context (multi-paragraph background), question (complex reasoning), answer (with evidence citations). Multi-hop QA dataset.",
            ],
            seed=args["seed"] + 1,
        )
        context_question_answer_args = args.copy()
        context_question_answer_args["max_inds"] = 100  # Override for this mode
        dfs.append(generator.generate_many(**context_question_answer_args))

    if mode in ["all", "answer_from_context_question"]:
        # Predict answer given context and question
        generator = GenericMultivariate(
            df=df,
            given_variables=["context", "question"],
            gen_variables=["answer"],
            name="hotpot_answer_from_context_question",
            descriptions=[
                "Given a context passage and question, provide the answer with supporting evidence from the text.",
                "Reading comprehension: answer multi-hop reasoning questions using information from the provided context.",
                "Given: context (background passage), question (multi-hop reasoning). Generate: answer (with evidence citations).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question_from_context_answer"]:
        # Predict question given context and answer
        generator = GenericMultivariate(
            df=df,
            given_variables=["context", "answer"],
            gen_variables=["question"],
            name="hotpot_question_from_context_answer",
            descriptions=[
                "Given a context passage and an answer, generate a multi-hop reasoning question that leads to this answer.",
                "Create complex questions that require multi-hop reasoning across the context to arrive at the given answer.",
                "Given: context (background passage), answer (with evidence). Generate: question (multi-hop reasoning that leads to this answer).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "context"]:
        # Predict just the context
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["context"],
            name="hotpot_context",
            descriptions=[
                "Generate multi-paragraph context passages suitable for complex reading comprehension questions.",
                "Create background information passages that contain interconnected facts for multi-hop reasoning tasks.",
                "Generate: context (multi-paragraph background with interconnected information for complex QA).",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'context_question_answer', 'answer_from_context_question', 'question_from_context_answer', 'context'"
        )

    return pd.concat(dfs, ignore_index=True)


def load_lcb_data(file_path: str | None = None):
    """Load LCB (Logic and Commonsense Benchmark) synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/lcb/lcb_gpt_4o_refine_70b_base_n=1000.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def generate_bare_lcb(mode: str = "all", **kwargs):
    """Generate LCB (Logic and Commonsense Benchmark) dataset with specified mode.

    Args:
        mode: One of "all", "question_answer", "answer_from_question", "question_from_answer", "question"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 300,  # Different base seed
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_total": 500,
        "max_inds": 50,
    }
    args.update(kwargs)

    df = load_lcb_data()
    dfs = []

    if mode in ["all", "question_answer"]:
        # Predict both question and answer
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["question", "answer"],
            name="lcb_question_answer",
            descriptions=[
                "Generate both logic and commonsense reasoning problems with their solutions. These cover algorithms, data structures, and logical reasoning.",
                "Create complete programming/logic problems: generate both the problem statement and step-by-step solution with reasoning.",
                "Generate: question (logic/programming problem), answer (detailed solution with reasoning). LCB benchmark problems.",
            ],
            seed=args["seed"] + 1,
        )
        question_answer_args = args.copy()
        question_answer_args["max_inds"] = 50  # Override for this mode
        dfs.append(generator.generate_many(**question_answer_args))

    if mode in ["all", "answer_from_question"]:
        # Predict answer given question
        generator = GenericMultivariate(
            df=df,
            given_variables=["question"],
            gen_variables=["answer"],
            name="lcb_answer_from_question",
            descriptions=[
                "Given a logic or programming problem, provide a step-by-step solution with clear reasoning.",
                "Solve logic and commonsense reasoning problems by providing detailed explanations and final answers.",
                "Given: question (logic/programming problem). Generate: answer (step-by-step solution with reasoning and final result).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question_from_answer"]:
        # Predict question given answer
        generator = GenericMultivariate(
            df=df,
            given_variables=["answer"],
            gen_variables=["question"],
            name="lcb_question_from_answer",
            descriptions=[
                "Given a detailed solution, generate a logic or programming problem that would lead to this solution.",
                "Create logic/programming problems that match the given step-by-step solution and reasoning.",
                "Given: answer (detailed solution with reasoning). Generate: question (logic/programming problem that leads to this solution).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question"]:
        # Predict just the question
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["question"],
            name="lcb_question",
            descriptions=[
                "Generate logic and commonsense reasoning problems covering algorithms, data structures, and programming concepts.",
                "Create diverse programming and logic problems suitable for technical assessment and reasoning evaluation.",
                "Generate: question (logic/programming problem statement only, without solution).",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'question_answer', 'answer_from_question', 'question_from_answer', 'question'"
        )

    return pd.concat(dfs, ignore_index=True)


def load_newsgroups_data(file_path: str | None = None):
    """Load Newsgroups synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/newsgroups/newsgroups_gpt_4o_refine_llama_70b_base_n=500.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def generate_bare_newsgroups(mode: str = "all", **kwargs):
    """Generate Newsgroups classification dataset with specified mode.

    Args:
        mode: One of "all", "newsgroup_message", "newsgroup_from_message", "message_from_newsgroup", "message"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 400,  # Different base seed
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_total": 500,
        "max_inds": 50,  # Smaller since only 500 total examples
    }
    args.update(kwargs)

    df = load_newsgroups_data()
    dfs = []

    if mode in ["all", "newsgroup_message"]:
        # Predict both newsgroup and message
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["newsgroup", "message"],
            name="newsgroups_newsgroup_message",
            descriptions=[
                "Generate both a newsgroup category and a message that belongs to that category. These are synthetic posts based on 20 Newsgroups dataset covering various topics.",
                "Create complete newsgroup posts: generate both the category (newsgroup) and the message content.",
                "Generate: newsgroup (category like sci.med, comp.graphics), message (post content). Text classification dataset.",
            ],
            seed=args["seed"] + 1,
        )
        newsgroup_message_args = args.copy()
        newsgroup_message_args["max_inds"] = 50  # Override for this mode
        dfs.append(generator.generate_many(**newsgroup_message_args))

    if mode in ["all", "newsgroup_from_message"]:
        # Predict newsgroup given message
        generator = GenericMultivariate(
            df=df,
            given_variables=["message"],
            gen_variables=["newsgroup"],
            name="newsgroups_newsgroup_from_message",
            descriptions=[
                "Given a message post, classify it into the appropriate newsgroup category (e.g., sci.med, comp.graphics, talk.politics).",
                "Text classification: determine which newsgroup category best fits the given message content.",
                "Given: message (post content). Generate: newsgroup (category classification).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "message_from_newsgroup"]:
        # Predict message given newsgroup
        generator = GenericMultivariate(
            df=df,
            given_variables=["newsgroup"],
            gen_variables=["message"],
            name="newsgroups_message_from_newsgroup",
            descriptions=[
                "Given a newsgroup category, generate a message that would be appropriate for that category.",
                "Create message content that fits the specified newsgroup topic (e.g., medical discussions for sci.med, graphics questions for comp.graphics).",
                "Given: newsgroup (category). Generate: message (post content appropriate for that category).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "message"]:
        # Predict just the message
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["message"],
            name="newsgroups_message",
            descriptions=[
                "Generate newsgroup message posts covering diverse topics like science, technology, politics, and recreation.",
                "Create message content in the style of online forum discussions and newsgroup posts.",
                "Generate: message (newsgroup post content only). Diverse online discussion posts.",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'newsgroup_message', 'newsgroup_from_message', 'message_from_newsgroup', 'message'"
        )

    return pd.concat(dfs, ignore_index=True)


def load_pubmed_data(file_path: str | None = None):
    """Load PubMed synthetic data from BARE dataset."""
    if file_path is None:
        file_path = "data/BARE/bare/sample_generated_data/pubmed/pubmed_gpt_4o_refine_llama_70b_base_n=1000.jsonl"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)

    return pd.DataFrame(data)


def generate_bare_pubmed(mode: str = "all", **kwargs):
    """Generate PubMed biomedical QA dataset with specified mode.

    Args:
        mode: One of "all", "context_question_answer", "answer_from_context_question", "question_from_context_answer", "context"
        **kwargs: Additional arguments passed to generate_many
    """
    args = {
        "seed": 500,  # Different base seed
        "n_per": -1,
        "n_iter": geom_and_poisson_iter(mean=32),
        "max_total": 500,
        "max_inds": 50,
    }
    args.update(kwargs)

    df = load_pubmed_data()
    dfs = []

    if mode in ["all", "context_question_answer"]:
        # Predict context, question, and answer
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["context", "question", "answer"],
            name="pubmed_context_question_answer",
            descriptions=[
                "Generate complete biomedical QA examples: context (medical research), question (clinical/research question), and answer (evidence-based response). These are synthetic medical questions based on PubMed literature.",
                "Create PubMed-style biomedical QA: generate context (research abstracts), question (medical/scientific), and answer (with evidence citations).",
                "Generate: context (biomedical research text), question (clinical question), answer (evidence-based response). Medical QA dataset.",
            ],
            seed=args["seed"] + 1,
        )
        context_question_answer_args = args.copy()
        context_question_answer_args["max_inds"] = 50  # Override for this mode
        dfs.append(generator.generate_many(**context_question_answer_args))

    if mode in ["all", "answer_from_context_question"]:
        # Predict answer given context and question
        generator = GenericMultivariate(
            df=df,
            given_variables=["context", "question"],
            gen_variables=["answer"],
            name="pubmed_answer_from_context_question",
            descriptions=[
                "Given biomedical research context and a clinical question, provide an evidence-based answer with supporting citations.",
                "Biomedical QA: answer medical/scientific questions using information from the provided research context.",
                "Given: context (biomedical research), question (clinical/scientific). Generate: answer (evidence-based response with citations).",
            ],
            seed=args["seed"] + 2,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "question_from_context_answer"]:
        # Predict question given context and answer
        generator = GenericMultivariate(
            df=df,
            given_variables=["context", "answer"],
            gen_variables=["question"],
            name="pubmed_question_from_context_answer",
            descriptions=[
                "Given biomedical research context and an answer, generate a clinical or scientific question that leads to this answer.",
                "Create medical/scientific questions that can be answered using the provided research context and evidence.",
                "Given: context (biomedical research), answer (evidence-based response). Generate: question (clinical/scientific question that leads to this answer).",
            ],
            seed=args["seed"] + 3,
        )
        dfs.append(generator.generate_many(**args))

    if mode in ["all", "context"]:
        # Predict just the context
        generator = GenericMultivariate(
            df=df,
            given_variables=[],
            gen_variables=["context"],
            name="pubmed_context",
            descriptions=[
                "Generate biomedical research abstracts and clinical study summaries suitable for medical question answering.",
                "Create scientific literature context covering medical research, clinical studies, and biomedical findings.",
                "Generate: context (biomedical research text for medical QA tasks).",
            ],
            seed=args["seed"] + 4,
        )
        dfs.append(generator.generate_many(**args))

    if not dfs:
        raise ValueError(
            f"Invalid mode: {mode}. Must be one of 'all', 'context_question_answer', 'answer_from_context_question', 'question_from_context_answer', 'context'"
        )

    return pd.concat(dfs, ignore_index=True)
