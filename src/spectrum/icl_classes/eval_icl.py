"""
e.g.
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --dataset dices --model_name google/gemma-3-1b-it --format colon --auto_restrict_texts
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --dataset novacomet_hypothesis --model_name google/gemma-3-1b-it --format colon
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --dataset novacomet_hypothesis --model_name tsor13/spectrum-Llama-3.1-8B-v0 --format spectrum
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --dataset habermas_question --model_name tsor13/spectrum-Llama-3.1-8B-v0 --format spectrum
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from spectrum.icl_classes.data_cache import load_tokenized_dataset
from spectrum.icl_classes.dataloaders import load_default_data
from spectrum.sample_utils import (
    MyDataCollatorForLanguageModeling,
    get_restrict_tokens,
    tokenize_example,
    tokenize_loss_texts,
)

global debug
debug = False


def load_eval_dataset(
    dataset_name,
    actual_model_name,
    cache_dir,
    max_length,
    dataset_kwargs,
    tokenizer,
    restrict_texts,
    eval_seed,
    max_eval_examples,
    auto_restrict_texts=False,
    # fixed_example_number # moved to dataset_kwargs
):
    val_dataset = load_tokenized_dataset(
        dataset_name,
        force_reload=False,
        # model_name=model_name,
        model_name=actual_model_name,
        cache_parent_dir=cache_dir,
        max_length=max_length,  # This now gets handled by the new caching system
        drop_additional_columns=False,
        save=True,
        **dataset_kwargs,
    )

    restrict_tokens = None
    if restrict_texts is not None:
        # get restrict tokens
        restrict_tokens = get_restrict_tokens(restrict_texts, tokenizer)

        # was running into error because need prior for restrict_labels to work properly
        val_dataset = val_dataset.map(
            restrict_labels,
            fn_kwargs={
                "tokenizer": tokenizer,
                "restrict_tokens": restrict_tokens,
            },
        )

    # val_dataset['dataset'] = dataset_name
    # change dataset name to dataset_name
    val_dataset = val_dataset.map(lambda x: {"dataset": dataset_name})
    print(dataset_name)

    # TAYLOR - maybe do some sort of filtering here for minimum / maximum number of examples? Or maybe we only want to include where there's exactly some number of examples?

    # Example index handling now happens in data_cache.add_example_inds.

    # if max_eval_examples is not None, shuffle deterministally (eval_seed) and truncate
    if max_eval_examples is not None:
        if max_eval_examples < len(val_dataset):
            val_dataset = val_dataset.select(
                np.random.RandomState(eval_seed).permutation(len(val_dataset))[
                    :max_eval_examples
                ]
            )
    return val_dataset


def evaluate_model(
    model,
    tokenizer,
    val_dataset,
    batch_size=2,
    max_length=1024,
    wandb_project=None,
    wandb_name=None,
    wandb_to_log=None,
    wandb_args=None,
    restrict_logprobs=None,
    fixed_example_number=None,
    auto_restrict_texts=False,
):
    """
    Evaluate a language model on a validation dataset, computing various metrics including loss, accuracy, and calibration.

    This function performs a comprehensive evaluation of a language model by:
    1. Processing the dataset in batches
    2. Computing token-level losses and accuracies
    3. Calculating calibration metrics (ECE)
    4. Tracking various performance metrics
    5. Optionally logging results to Weights & Biases

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The language model to evaluate
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer for the model
    val_dataset : datasets.Dataset
        The validation dataset to evaluate on
    batch_size : int, optional
        Batch size for evaluation, by default 2
    max_length : int, optional
        Maximum sequence length, by default 1024
    wandb_project : str, optional
        Weights & Biases project name for logging. If None, no logging is performed.
    wandb_args : dict, optional
        Additional arguments to pass to wandb.init(), by default None
    restrict_logprobs : list, optional
        List of token IDs to restrict log probabilities to, by default None

    Returns
    -------
    dict
        A dictionary containing:
        - raw_results: pd.DataFrame
            Detailed results for each instance including loss, accuracy, and log probabilities
        - results: pd.DataFrame
            Aggregated results grouped by example and dataset, including:
            - mean loss
            - mean first token accuracy
            - concatenated correct and incorrect log probabilities
            - mean number of tokens
            - number of instances
            - Expected Calibration Error (ECE) metrics
        - wandb_run: wandb.Run or None
            The Weights & Biases run object if logging was performed, otherwise None.
    """
    data_collator = MyDataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    data_losses = []
    for batch_ind in tqdm(
        range(0, len(val_dataset), batch_size), total=len(val_dataset) // batch_size
    ):
        batch = val_dataset[batch_ind : batch_ind + batch_size]
        # change format to list of dicts
        keys = list(batch.keys())
        this_batch_size = len(batch[keys[0]])
        new_batch = [
            {key: batch[key][j] for key in keys} for j in range(this_batch_size)
        ]
        processed_batch = data_collator(new_batch)
        # run through model
        with torch.no_grad():
            outputs = model(
                input_ids=processed_batch["input_ids"].to(model.device),
                attention_mask=processed_batch["attention_mask"].to(model.device),
            )
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        batch_restrict_tokens = None

        if "restrict_tokens" in batch.keys():
            batch_restrict_tokens = batch["restrict_tokens"]
            for i, restrict_tokens in enumerate(batch_restrict_tokens):
                # set logprobs to -inf for all except the restricted logprobs
                logit_mask = torch.ones(shift_logits.shape[-1]).to(shift_logits.device)
                # set to zero for restricted logprobs
                logit_mask[restrict_tokens] = 0
                shift_logits[i] = shift_logits[i] + logit_mask * -100
                # assert that labels are all in restrict_tokens
                if not np.all(
                    [
                        t in restrict_tokens + [-100]
                        for t in processed_batch["labels"][i]
                    ]
                ):
                    raise ValueError("Labels are not all in restrict_tokens")

        if restrict_logprobs is not None:
            # set logprobs to -inf for all except the restricted logprobs
            logit_mask = torch.ones(shift_logits.shape[-1]).to(shift_logits.device)
            # set to zero for restricted logprobs
            logit_mask[restrict_logprobs] = 0
            shift_logits = shift_logits + logit_mask.unsqueeze(0) * -100

        shift_labels = processed_batch["labels"].to(model.device)[..., 1:].contiguous()
        guesses = shift_logits.argmax(dim=-1)

        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_loss = token_loss.view(shift_logits.shape[0], shift_logits.shape[1])

        data_ids = processed_batch["data_ids"].to(model.device)
        shift_data_ids = data_ids[..., 1:].contiguous()

        example_inds = processed_batch["example_inds"].to(model.device)
        shift_example_inds = example_inds[..., 1:].contiguous()

        # get shift_logprobs by doing softmax on last dim
        shift_logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        label_mask = shift_labels != -100
        min_p_calibration = 0.01
        min_logp_calibration = np.log(min_p_calibration)
        actual_logprobs = shift_logprobs[label_mask]

        for i in range(len(example_inds)):
            # n_datapoints = example_inds[i].max() + 1
            # if no shift_labels, skip
            if (shift_labels[i] == -100).all():
                print("No shift_labels for example ", i)
                continue
            n_datapoints = shift_example_inds[i][shift_labels[i] != -100].max() + 1
            for j in range(n_datapoints):
                # get the losses where data_ids == j
                # data_inds = (shift_data_ids[i] == j)
                # get the losses where data_ids == j and labels are not -100
                example_indices = (shift_example_inds[i] == j) & (
                    shift_labels[i] != -100
                )
                if example_indices.sum() == 0:
                    print("No example indices...")
                    this_input_ids = processed_batch["input_ids"][i][1:][
                        shift_example_inds[i].cpu() == j
                    ]
                    print(tokenizer.decode(this_input_ids))
                    print(shift_labels[i, shift_example_inds[i] == j])
                    # if j is 0, just raise a warning - could be because no prior
                    if j == 0:
                        print(
                            "WARNING: No example indices for j = 0. Is there a prior?"
                        )
                        # continue in j
                        continue
                    else:
                        raise ValueError("No example indices")
                first_ind = (
                    1 * example_indices
                ).argmax()  # TODO might be first or last, not sure

                this_logprobs = shift_logprobs[i, example_indices]
                this_labels = shift_labels[i, example_indices]
                correct_logprobs = this_logprobs[
                    torch.arange(len(this_logprobs)), this_labels
                ]
                # get all logprobs that are not the correct logprob
                incorrect_logprobs = this_logprobs.clone()
                # set the correct logprob to -inf
                incorrect_logprobs[torch.arange(len(this_logprobs)), this_labels] = (
                    -float("inf")
                )
                # get the max incorrect logprob
                # get a list of all incorrect logprobs above min_logp_calibration
                incorrect_logprobs_above_min = incorrect_logprobs[
                    incorrect_logprobs > min_logp_calibration
                ]
                # get the mean of the incorrect logprobs

                restricted_logprobs = None
                if batch_restrict_tokens is not None:
                    restricted_logprobs = shift_logprobs[i, example_indices]
                    restricted_logprobs = restricted_logprobs[
                        :, batch_restrict_tokens[i]
                    ]
                    restricted_logprobs = restricted_logprobs.detach().cpu().numpy()

                # ALL TOKENS CORRECT? all_acc
                data_losses.append(
                    {
                        "loss": token_loss[i, example_indices].sum().cpu().item(),
                        "instance_ind": batch_ind + i,
                        "example_ind": j,
                        "n_tokens": example_indices.sum().cpu().item(),
                        "first_acc": (
                            1 * (guesses[i, first_ind] == shift_labels[i, first_ind])
                        )
                        .cpu()
                        .item(),
                        "acc": (
                            1
                            * (
                                guesses[i, example_indices]
                                == shift_labels[i, example_indices]
                            )
                        )
                        .cpu()
                        .float()
                        .mean()
                        .item(),
                        "all_acc": (
                            1
                            * (
                                guesses[i, example_indices]
                                == shift_labels[i, example_indices]
                            )
                        )
                        .cpu()
                        .float()
                        .min()
                        .item(),
                        "correct_logprobs": correct_logprobs.detach().cpu().numpy(),
                        "incorrect_logprobs": incorrect_logprobs_above_min.detach()
                        .cpu()
                        .numpy(),
                        "dataset": batch["dataset"][i],
                        "data_id": shift_data_ids[i, shift_example_inds[i] == j][0]
                        .cpu()
                        .item(),
                        "restricted_logprobs": restricted_logprobs,
                    }
                )

    # to dataframe
    data_losses_df = pd.DataFrame(data_losses)

    if fixed_example_number is not None:
        # only keep examples where example_ind is less than fixed_example_number
        data_losses_df = data_losses_df[
            data_losses_df["example_ind"] <= fixed_example_number
        ]

    # group by instance_ind and get loss and mean
    # for correct_logprobs, keep running list
    grouped = data_losses_df.groupby(["example_ind", "dataset"]).aggregate(
        {
            "loss": "mean",
            "first_acc": "mean",
            "acc": "mean",
            "all_acc": "mean",
            "correct_logprobs": lambda x: np.concatenate(list(x)),
            "incorrect_logprobs": lambda x: np.concatenate(list(x)),
            "n_tokens": "mean",
            "instance_ind": "count",
        }
    )

    def get_ece_stats(correct_logprobs, incorrect_logprobs, thresh=0.01, n_bins=10):
        log_thresh = np.log(thresh)
        correct_logprobs_above_thresh = correct_logprobs[correct_logprobs > log_thresh]
        incorrect_logprobs_above_thresh = incorrect_logprobs[
            incorrect_logprobs > log_thresh
        ]
        all_logprobs = np.concatenate(
            [correct_logprobs_above_thresh, incorrect_logprobs_above_thresh]
        )
        n_bins = min(n_bins, len(all_logprobs))
        # get bins by percentile every n_bins+1
        bin_cutoffs = np.percentile(all_logprobs, np.linspace(0, 100, n_bins + 1))
        # get mean conf in each bin
        bin_means = []
        bin_p_correct = []
        for bin_min, bin_max in zip(bin_cutoffs[:-1], bin_cutoffs[1:]):
            bin_inds = (all_logprobs >= bin_min) & (all_logprobs <= bin_max)
            if bin_inds.sum() > 0:
                bin_means.append(np.mean(np.exp(all_logprobs[bin_inds])))
                n_all = bin_inds.sum()
                n_correct = (
                    (correct_logprobs >= bin_min) & (correct_logprobs <= bin_max)
                ).sum()
                bin_p_correct.append(n_correct / n_all)
        bin_means = np.array(bin_means)
        bin_p_correct = np.array(bin_p_correct)
        ece = np.abs(bin_means - bin_p_correct).mean()
        return {
            "ece": ece.item(),
            "bin_means": bin_means,
            "bin_p_correct": bin_p_correct,
            "n_bins": n_bins,
            "n_all": len(all_logprobs),
            "n_correct": len(correct_logprobs_above_thresh),
        }

    # rename instance_ind to n_instances
    grouped = grouped.rename(columns={"instance_ind": "n_instances"})

    # apply to each row in grouped
    grouped["ece"] = grouped.apply(
        lambda row: get_ece_stats(row["correct_logprobs"], row["incorrect_logprobs"]),
        axis=1,
    )

    wandb_run = None
    if wandb_project is None:
        # make dataset + _eval1
        wandb_project = f"{val_dataset[0]['dataset']}_icl_eval_fixed"
        if auto_restrict_texts:
            wandb_project += "_restrict"
    if wandb_project is not None:
        # log grouped to wandb
        wandb_config = wandb_args or {}
        wandb_config["model_name"] = model.name_or_path
        wandb_config["batch_size"] = batch_size
        wandb_config["fixed_example_number"] = fixed_example_number
        wandb_config["auto_restrict_texts"] = auto_restrict_texts
        # format
        # wandb_config["format"] = format
        wandb.login()  # ensures the key from env is used
        if wandb_name is None:
            wandb_name = f"{model.name_or_path}"
        wandb_run = wandb.init(
            project=wandb_project,
            # entity="tsor1313",           # <-- your W&B username/org
            # name="gpt2-models",         # optional run name
            name=wandb_name,
            # reinit=True,
            config=wandb_config,
        )

        # log eval/loss and first_acc
        wandb.log({"eval/loss": data_losses_df["loss"].mean()})
        wandb.log({"eval/loss_sem": data_losses_df["loss"].sem()})
        wandb.log({"eval/first_acc": data_losses_df["first_acc"].mean()})
        wandb.log({"eval/first_acc_sem": data_losses_df["first_acc"].sem()})
        wandb.log({"eval/acc": data_losses_df["acc"].mean()})
        wandb.log({"eval/acc_sem": data_losses_df["acc"].sem()})
        wandb.log({"eval/all_acc": data_losses_df["all_acc"].mean()})
        wandb.log({"eval/all_acc_sem": data_losses_df["all_acc"].sem()})

        subset_geq1 = data_losses_df[data_losses_df["example_ind"] >= 1]
        wandb.log({"eval/loss_geq_1": subset_geq1["loss"].mean()})
        wandb.log({"eval/loss_geq_1_sem": subset_geq1["loss"].sem()})
        wandb.log({"eval/first_acc_geq_1": subset_geq1["first_acc"].mean()})
        wandb.log({"eval/first_acc_geq_1_sem": subset_geq1["first_acc"].sem()})
        wandb.log({"eval/acc_geq_1": subset_geq1["acc"].mean()})
        wandb.log({"eval/acc_geq_1_sem": subset_geq1["acc"].sem()})
        wandb.log({"eval/all_acc_geq_1": subset_geq1["all_acc"].mean()})
        wandb.log({"eval/all_acc_geq_1_sem": subset_geq1["all_acc"].sem()})

        # do k
        subset_k = subset_geq1[
            subset_geq1["example_ind"] == subset_geq1["example_ind"].max() - 1
        ]
        wandb.log({"eval/loss_k": subset_k["loss"].mean()})
        wandb.log({"eval/loss_k_sem": subset_k["loss"].sem()})
        wandb.log({"eval/first_acc_k": subset_k["first_acc"].mean()})
        wandb.log({"eval/first_acc_k_sem": subset_k["first_acc"].sem()})
        wandb.log({"eval/acc_k": subset_k["acc"].mean()})
        wandb.log({"eval/acc_k_sem": subset_k["acc"].sem()})
        wandb.log({"eval/all_acc_k": subset_k["all_acc"].mean()})
        wandb.log({"eval/all_acc_k_sem": subset_k["all_acc"].sem()})

        # log statistics (len dataset)
        wandb.log({"statistics/val_len": len(val_dataset)})

        if wandb_to_log is not None:
            for key in wandb_to_log.keys():
                wandb.log({key: wandb_to_log[key]})

        all_correct_logprobs = np.concatenate(data_losses_df["correct_logprobs"].values)
        all_incorrect_logprobs = np.concatenate(
            data_losses_df["incorrect_logprobs"].values
        )
        ece_stats = get_ece_stats(all_correct_logprobs, all_incorrect_logprobs)
        for key, value in ece_stats.items():
            wandb.log({f"calibration/{key}": value})

        geq1_correct_logprobs = np.concatenate(subset_geq1["correct_logprobs"].values)
        geq1_incorrect_logprobs = np.concatenate(
            subset_geq1["incorrect_logprobs"].values
        )
        geq1_ece_stats = get_ece_stats(geq1_correct_logprobs, geq1_incorrect_logprobs)
        for key, value in geq1_ece_stats.items():
            wandb.log({f"calibration/geq1.{key}": value})

        k_correct_logprobs = np.concatenate(subset_k["correct_logprobs"].values)
        k_incorrect_logprobs = np.concatenate(subset_k["incorrect_logprobs"].values)
        k_ece_stats = get_ece_stats(k_correct_logprobs, k_incorrect_logprobs)
        for key, value in k_ece_stats.items():
            wandb.log({f"calibration/k.{key}": value})

        def log_table_with_name(table, name):
            wandb_table = table
            # reset index
            wandb_table = wandb_table.reset_index()
            # convert to dict
            wandb_table = wandb_table.to_dict(orient="records")
            wandb_rows = []
            for row in wandb_table:
                row_dict = {}
                for key in row.keys():
                    row_dict[key] = row[key]
                wandb_rows.append(row_dict)

            def clean_type(val):
                if isinstance(val, np.float64):
                    return float(val)
                elif isinstance(val, np.int64):
                    return int(val)
                elif isinstance(val, np.ndarray):
                    return val.tolist()
                elif isinstance(val, dict):
                    raise ValueError("Dicts are not allowed")
                else:
                    return val

            # get dict
            for row in wandb_table:
                row_dict = {}
                for key in row.keys():
                    if isinstance(row[key], dict):
                        for subkey in row[key].keys():
                            # row_dict[f"{key}/{subkey}"] = row[key][subkey]
                            row_dict[f"{key}/{subkey}"] = clean_type(row[key][subkey])
                    else:
                        row_dict[key] = clean_type(row[key])
                wandb_rows.append(row_dict)
            wandb_table = pd.DataFrame(wandb_rows)
            wandb.log({name: wandb_table})

        # drop  the following from grouped
        grouped = grouped.drop(columns=["correct_logprobs", "incorrect_logprobs"])
        log_table_with_name(grouped, "results")

        # keep only example_ind, instance_ind, the loss, acc, first_acc, all_acc, ece, and n_tokens in data_losses_df
        to_log = data_losses_df[
            [
                "example_ind",
                "instance_ind",
                "loss",
                "acc",
                "first_acc",
                "all_acc",
                "n_tokens",
            ]
        ]
        log_table_with_name(to_log, "data_losses")

    return {
        "raw_results": data_losses_df,
        "results": grouped,
        "wandb_run": wandb_run,
    }


def get_example_k(
    dataset_name,
    actual_model_name,
    cache_dir,
    dataset_kwargs,
    tokenizer,
    restrict_texts,
    eval_seed,
    max_eval_examples,
    max_length=None,
    # fixed_example_number,
    format_k="spectrum",  # the format to use for choosing length
    model_select_k="google/gemma-3-1b-it",  # the model to use for selecting k
    target_min_seqs=200,
    # target_exclude_percentile = 25,
    target_exclude_percentile=50,  # remove the bottom x% of examples
    # target_exclude_percentile = 75, # remove the bottom x% of examples
    # target_exclude_percentile = 50, # remove the bottom x% of examples
    target_min_k=2,
    select_max_length=2048,
):
    if max_length is not None:
        print(
            "Using select_max_length: for selecting k, not max length: ",
            select_max_length,
        )
    dataset_kwargs_copy = dataset_kwargs.copy()
    dataset_kwargs_copy["format"] = format_k
    # read in the data for the dataset with format_k and max_length
    dataset_kwargs_copy["max_length"] = select_max_length
    # remove fixed_example_number if it exists
    if "fixed_example_number" in dataset_kwargs_copy.keys():
        del dataset_kwargs_copy["fixed_example_number"]
    val_dataset = load_tokenized_dataset(
        dataset_name,
        force_reload=False,
        model_name=model_select_k,
        cache_parent_dir=cache_dir,
        # max_length=select_max_length,
        **dataset_kwargs_copy,
    )

    def get_max_example_ind(example):
        # mask example_inds where labels are -100, get the highest example_ind
        max_ind = np.array(example["example_inds"])[example["labels"] != -100].max()
        # if last label is not -100, then subtract one because it's not complete
        if example["labels"][-1] != -100:
            max_ind -= 1
        return {"max_example_ind": int(max_ind)}

    val_dataset = val_dataset.map(get_max_example_ind)
    print("No filtering of examples based on fixed_example_number")
    print(
        f"Percentiles of max_example_ind (every 25%): {np.percentile(val_dataset['max_example_ind'], [0, 25, 50, 75, 100])}"
    )

    best_k = -1
    best_val = -1
    for k in range(1, np.max(val_dataset["max_example_ind"]) + 1):
        val = ((np.array(val_dataset["max_example_ind"]) >= k) * k).sum()
        if val > best_val:
            best_val = val
            best_k = k
    k = best_k
    print(f"Best k: {best_k}, best n: {best_val}")

    this_inds = np.argwhere(np.array(val_dataset["max_example_ind"]) >= k).flatten()
    orig_inds = np.array(val_dataset["original_index"])
    keep_inds = orig_inds[this_inds]
    if k <= 1:
        # if at least 10 instances of min_k, then set k to min_k
        if np.sum(np.array(val_dataset["max_example_ind"]) >= target_min_k) >= 10:
            k = target_min_k
        else:
            raise ValueError("Not enough examples to select k")
    return {
        "keep_inds": keep_inds,
        "k": int(k),
    }


def completion_logprobs(model, tokenizer, val_dataset, batch_size=2):
    """
    Given a val dataset, get the summed logprob of the completion for each example. Also keep track of the number of tokens in the completion.
    """
    data_collator = MyDataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    completion_logprobs = []

    for batch_ind in tqdm(
        range(0, len(val_dataset), batch_size), total=len(val_dataset) // batch_size
    ):
        batch = val_dataset[batch_ind : batch_ind + batch_size]
        # change format to list of dicts
        keys = list(batch.keys())
        this_batch_size = len(batch[keys[0]])
        new_batch = [
            {key: batch[key][j] for key in keys} for j in range(this_batch_size)
        ]
        processed_batch = data_collator(new_batch)

        # run through model
        with torch.no_grad():
            outputs = model(
                input_ids=processed_batch["input_ids"].to(model.device),
                attention_mask=processed_batch["attention_mask"].to(model.device),
            )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = processed_batch["labels"].to(model.device)[..., 1:].contiguous()

        # get log probabilities
        shift_logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # get the log probabilities of the correct tokens
        correct_logprobs = shift_logprobs[
            torch.arange(shift_logprobs.shape[0]).unsqueeze(1),
            torch.arange(shift_logprobs.shape[1]).unsqueeze(0),
            shift_labels,
        ]

        # mask out padding tokens
        label_mask = shift_labels != -100
        masked_logprobs = correct_logprobs * label_mask

        # sum log probabilities for each example
        summed_logprobs = masked_logprobs.sum(dim=1)
        n_tokens = label_mask.sum(dim=1)

        for i in range(len(processed_batch["data_ids"])):
            completion_logprobs.append(
                {
                    "instance_ind": batch_ind + i,
                    "summed_logprob": summed_logprobs[i].cpu().item(),
                    "n_tokens": n_tokens[i].cpu().item(),
                }
            )

    # convert to dataframe
    completion_logprobs_df = pd.DataFrame(completion_logprobs)

    return completion_logprobs_df


def restrict_labels(example, tokenizer, restrict_tokens=None):
    labels = example["labels"]
    if restrict_tokens is None:
        if "restrict_tokens" not in example.keys():
            raise ValueError("Restrict tokens not found in example")
        restrict_tokens = example["restrict_tokens"]
    labels = np.where(np.isin(labels, restrict_tokens), labels, -100)
    if np.all(labels == -100):
        raise ValueError("All labels are -100")
    # total examples is max example_ind where labels is not -100
    total_examples = np.max(np.array(example["example_inds"])[labels != -100]) + 1
    if (
        np.sum(labels != -100) < total_examples
    ):  # TAYLOR - for now, allowing a fudge factor of 1 for the last example if it's truncated partially
        print("Total examples: ", total_examples)
        print("Sum of labels not -100: ", np.sum(labels != -100))
        original_unique_labels = np.unique(example["labels"])
        # remove -100
        original_unique_labels = original_unique_labels[original_unique_labels != -100]
        # get unique labels not in restrict_tokens
        unique_labels_not_in_restrict_tokens = np.setdiff1d(
            original_unique_labels, restrict_tokens
        )
        print("unique_labels_not_in_restrict_tokens")
        for t in unique_labels_not_in_restrict_tokens:
            print(f"{t} : __{tokenizer.decode([t])}__")
        # also print restrict_tokens
        print("restrict_tokens")
        for t in restrict_tokens:
            print(f"{t} : __{tokenizer.decode([t])}__")
        raise ValueError("Not enough labels to match the number of examples")
    return {
        "labels": labels,
    }


def evaluate_model_on_dataset(
    model_name,
    dataset_name,
    max_length,
    dataset_kwargs={},
    batch_size=32,
    restrict_texts=None,
    wandb_project=None,
    wandb_name=None,
    wandb_args=None,
    tokenizer_path=None,
    # New parameters for automatic k-fold model loading
    # auto_kfold=False,
    # base_model_name=None,
    # kfold=4,
    output_root="models_v6",
    # kfold_seed=42,
    max_eval_examples=None,
    auto_fixed_examples=False,
    eval_seed=42,
    fixed_example_number=None,
    cache_dir="tokenized_eval_datasets",
    ignore_prior=False,
    auto_restrict_texts=False,
):
    # # Automatic k-fold model loading
    # if auto_kfold:
    #     if base_model_name is None:
    #         raise ValueError("base_model_name must be provided when auto_kfold=True")
    #
    #     print(f"Auto-loading k-fold model for dataset '{dataset_name}' from base model '{base_model_name}'")
    # #
    #     # Extract formatting flags from dataset_kwargs
    #     format = dataset_kwargs.get('format', 'spectrum')
    #     is_spectrum = dataset_kwargs.get('is_spectrum', False)
    #     is_chat = dataset_kwargs.get('is_chat', False)
    #
    #     try:
    #         model, tokenizer, model_path = load_kfold_model_for_evaluation(
    #             model_name=base_model_name,
    #             dataset_name=dataset_name,
    #             kfold=kfold,
    #             output_root=output_root,
    #             format=format,
    #             is_spectrum=is_spectrum,
    #             is_chat=is_chat,
    #             kfold_seed=kfold_seed,
    #             device_map="auto"
    #         )
    #
    #         print(f"Successfully loaded model from: {model_path}")
    #         # Update model_name for wandb logging
    #         actual_model_name = model_path
    #
    #     except Exception as e:
    #         print(f"Failed to auto-load k-fold model: {e}")
    #         print("Available models:")
    #         available = get_available_models(output_root)
    #         for name, info in available.items():
    #             print(f"  {name}: {info['configurations']}")
    #         raise
    # else:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    actual_model_name = model_name

    # Override tokenizer if separate tokenizer_path is provided
    if tokenizer_path is not None:
        print(f"Loading separate tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

    args = {
        "permutations_per_iter": 1,
        "include_prior_prob": (
            0 if ignore_prior else 1
        ),  # For eval, generally want to include the prior so loss is calculated on first example as well.
        "include_data_formatter_description_prob": 0,
        "include_replacement_prob": 0,
        "include_tags_prob": 0,
    }
    if dataset_kwargs:
        print(
            "Unfortunately, dataset_kwargs are overridden by data cache if precached exists"
        )
    args.update(dataset_kwargs)

    # Make sure tokenizer is available for all template formatters
    args["tokenizer"] = tokenizer

    # Modify cache directory if ignoring prior
    if ignore_prior:
        cache_dir = cache_dir + "_no_prior"

    # load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )

    # load the eval dataset
    # def load_eval_dataset(dataset_name, actual_model_name, cache_dir, max_length, dataset_kwargs, tokenizer, restrict_texts):
    eval_dataset_args = {
        "dataset_name": dataset_name,
        "actual_model_name": actual_model_name,
        "cache_dir": cache_dir,
        "max_length": max_length,
        "dataset_kwargs": args,
        "tokenizer": tokenizer,
        "restrict_texts": restrict_texts,
        "eval_seed": eval_seed,
        "max_eval_examples": max_eval_examples,
        # "auto_restrict_texts": auto_restrict_texts,
        # "fixed_example_number": fixed_example_number,
    }

    if auto_fixed_examples:
        example_k_args = get_example_k(**eval_dataset_args)
        eval_dataset_args["dataset_kwargs"]["keep_inds"] = example_k_args["keep_inds"]
        eval_dataset_args["dataset_kwargs"]["fixed_example_number"] = example_k_args[
            "k"
        ]
        eval_dataset_args["max_length"] = None
        fixed_example_number = example_k_args["k"]
    elif fixed_example_number is not None:
        eval_dataset_args["dataset_kwargs"][
            "fixed_example_number"
        ] = fixed_example_number
    # eval_dataset_args['auto_restrict_texts'] = auto_restrict_texts

    val_dataset = load_eval_dataset(
        **eval_dataset_args,
        # auto_restrict_texts=auto_restrict_texts,
    )

    restrict_tokens = None
    if restrict_texts is not None:
        restrict_tokens = get_restrict_tokens(restrict_texts, tokenizer)

    if auto_restrict_texts:
        if restrict_texts is not None:
            raise ValueError(
                "restrict_texts must be None when auto_restrict_texts is True"
            )
        # get unique labels
        unique_labels = set()
        for example in val_dataset:
            unique_labels.update(example["labels"])
        # get unique labels that are not -100
        unique_labels = unique_labels - {-100}
        # TAYLOR - there's a tradeoff here. You could either decide to 1) include the EOS/\n tokens, but this means that you may get lower acc vs forcing the model to know to start w/ the outputs (e.g. on polis_vote, often predicts \n over Agree/Disagree for colon).
        # or 2) you could omit EOS/\n tokens, but this would be a problem when you have sequences which are a subset of one another (e.g., lewidi ven, annotators assign multiple labels to one example).
        # I'm opting for 1 for now.
        # A third option would be to look at the labels and check if the terminal token is necessary for distinguishing outputs - but this is too much work haha.
        # Then again, this isn't so bad as long as you do at least 1-shot (which is what I'm reporting for the paper anyways. It's mostly a problem at the 0th example).
        restrict_tokens = list(unique_labels)
        # to tensor
        restrict_tokens = torch.tensor(restrict_tokens)
        max_expected_unique = 30
        if len(unique_labels) > max_expected_unique:
            print(
                f"WARNING: {len(unique_labels)} unique labels found, expected {max_expected_unique}"
            )
            raise ValueError("Too many unique labels")
        restrict_texts = tokenizer.batch_decode(restrict_tokens.reshape(-1, 1))
        # tokenize restrict tokens
        print(f"Tokenized restrict tokens: {restrict_texts}")
        # get restrict tokens
        # restrict_tokens = get_restrict_tokens(unique_labels, tokenizer)

    # evaluate
    out = evaluate_model(
        model,
        tokenizer,
        # data,
        val_dataset,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        wandb_args={
            "model_name": actual_model_name,
            # "base_model_name": base_model_name if auto_kfold else None,
            # "auto_kfold": auto_kfold,
            # "kfold": kfold if auto_kfold else None,
            # "restrict_tokens": restrict_tokens,
            "restrict_texts": restrict_texts,
            "max_length": max_length,
            "wandb_name": wandb_name,
            "is_chat": dataset_kwargs.get("is_chat", False),
            "is_spectrum": dataset_kwargs.get("is_spectrum", False),
            "auto_fixed_examples": auto_fixed_examples,
            "fixed_example_number": fixed_example_number,
            "format": dataset_kwargs["format"],
            "auto_restrict_texts": auto_restrict_texts,
        },
        restrict_logprobs=restrict_tokens,
        wandb_to_log={"statistics/before_max_eval_len": len(val_dataset)},
        batch_size=batch_size,
        max_length=max_length,
        fixed_example_number=fixed_example_number,
        auto_restrict_texts=auto_restrict_texts,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="spectrum model path (for manual loading)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional path to a different tokenizer. If not specified, uses the model's tokenizer.",
    )
    # dataset name (required)
    parser.add_argument("--dataset", type=str, required=True)
    # batch size
    parser.add_argument("--batch_size", type=int, default=8)
    # add wandb project
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    # dataset kwargs
    parser.add_argument("--dataset_kwargs", type=str, default=None)
    # restrict texts (list of strings)
    parser.add_argument("--restrict_texts", type=str, default=None)
    # evaluation limits
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffling when limiting eval examples",
    )
    # Fixed example number for filtering
    parser.add_argument(
        "--fixed_example_number",
        type=int,
        default=None,
        help="Minimum number of examples required per instance",
    )
    # Cache directory
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="tokenized_eval_datasets",
        help="Directory for tokenized dataset cache (default: tokenized_eval_datasets)",
    )
    # Prior probability flag
    parser.add_argument(
        "--ignore_prior",
        action="store_true",
        help="Set include_prior_prob to 0 and append 'no_prior' to cache directory",
    )
    # Template format argument (same as sft.py)
    parser.add_argument(
        "--format",
        dest="format",
        type=str,
        required=True,
        choices=["chat", "spectrum", "colon"],
        help="Template format to use. 'chat' uses user/assistant roles, others use description/input/output structure.",
    )
    # debug flag
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Automatic k-fold model loading
    parser.add_argument(
        "--output_root",
        type=str,
        default="models_v6",
        help="Root directory for trained models",
    )

    # add auto fixed examples
    parser.add_argument(
        "--auto_fixed_examples", action="store_true", help="Automatically select k"
    )

    # add auto restrict texts
    parser.add_argument(
        "--auto_restrict_texts",
        action="store_true",
        help="Automatically restrict texts",
    )

    args = parser.parse_args()

    debug = args.debug

    # # Validation: either manual model_name or auto_kfold must be specified
    # if not args.auto_kfold and not args.model_name:
    #     parser.error("Either --model_name or --auto_kfold (with --base_model_name) must be specified")

    # if args.auto_kfold and not args.base_model_name:
    #     parser.error("--base_model_name is required when using --auto_kfold")

    if args.restrict_texts is not None:
        args.restrict_texts = eval(args.restrict_texts)
    # parse dataset kwargs into dict
    if args.dataset_kwargs is not None:
        args.dataset_kwargs = eval(args.dataset_kwargs)
    else:
        args.dataset_kwargs = {}

    # Add template formatting flags to dataset_kwargs (same pattern as sft.py)
    args.dataset_kwargs["format"] = args.format

    evaluate_model_on_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset,
        dataset_kwargs=args.dataset_kwargs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        restrict_texts=args.restrict_texts,
        tokenizer_path=args.tokenizer_path,
        # New auto k-fold parameters
        # auto_kfold=args.auto_kfold,
        # base_model_name=args.base_model_name,
        # kfold=args.kfold,
        output_root=args.output_root,
        # kfold_seed=args.kfold_seed,
        max_eval_examples=args.max_eval_examples,
        eval_seed=args.eval_seed,
        fixed_example_number=args.fixed_example_number,
        cache_dir=args.cache_dir,
        ignore_prior=args.ignore_prior,
        auto_fixed_examples=args.auto_fixed_examples,
        auto_restrict_texts=args.auto_restrict_texts,
    )
