import json
import os
import re

# Import chat utilities
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# import inspect


class ICLClass(ABC):
    def __init__(
        self,
        name: str,
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        length: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        # replacement: bool = True,
        replacement: bool = False,
    ):
        """
        Base class for ICL classes.
        :param name: Identifier for this random class.
        :param descriptions: description distributions (list of str or dict of str to str) or None.
        """
        if name is None:
            name = "ICLClass"
        self.name = name
        self.descriptions = descriptions
        if length is None:
            length = 1
        self.length = length
        self.additional_params = additional_params
        self.replacement = replacement

    def __len__(self):
        return self.length

    def verify_ind(self, ind: int | None = None):
        if ind is None:
            if self.length > 1:
                raise ValueError(
                    f"Random class {self.name} has length {self.length}, so index must be provided."
                )
        elif ind >= self.length:
            raise ValueError(
                f"Index {ind} is out of bounds for random class {self.name} with length {self.length}."
            )

    @abstractmethod
    def get_raw_samples(
        self,
        seed: int,
        max_n: int,
        ind: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate textual representations.
        :param seed: Random seed.
        :param max_n: Maximum number of texts to generate.
        :param include_description: Whether to include descriptions in the output.
        :return: List of generated text strings, with attributes "text" and "compute_loss" and optionally "data_id"
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def samples_to_messages(
        self,
        samples: List[Dict[str, Any]],
        seed: int | None = None,
        include_description: bool = True,
        permute_seed: int | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convert samples to texts.
        """

        if permute_seed is not None:
            samples = samples.copy()
            np.random.seed(permute_seed)
            np.random.shuffle(samples)

        messages = []
        if include_description:  # randomly choose one of the descriptions
            # randomly select a description, making deterministic with the seed
            np.random.seed(seed)
            description = np.random.choice(self.descriptions)
            messages.append({"role": "description", "content": description})
        # with
        for example_ind, sample in enumerate(samples):
            # randomly set
            sample_messages = self.sample_to_messages(sample, example_ind)
            messages.extend(sample_messages)
        return messages

    def sample_to_messages(
        self,
        sample: Dict[str, Any],
        example_ind: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("not yet implemented, new")

    def generate_many(
        self,
        seed: int,
        n_per: int = 1,
        n_iter: int | Iterable[int] | None = None,
        max_inds: int | None = None,
        max_total: int | None = None,
        permutations_per_iter: int = 1,
        include_description_prob: float = 0.6,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate data.
        n_per: number of samples per datatype. If -1, generate ALL samples from each instance without repeats.
        n_iter: the maximum number of samples to generate. If there aren't many samples, can leave this None. It can either be an int, or a generator function (eg if you want to randomly choose the number of samples to include).
        max_inds: If there are many samples (e.g., individuals) can limit the number of people to include.
        max_total: Maximum total number of output rows. If None, no limit applied.
        permutations_per_iter: If there are many permutations, can limit the number of permutations to include.
        include_description_prob: Probability of including a description.
        include_data_formatter_description_prob: Probability of including a data formatter description.
        include_replacement_prob: Probability of including a replacement.
        include_tags_prob: Probability of including tags.
        vary_formatter: Whether to vary the formatter.
        """
        n_inds = self.length
        if max_inds is not None:
            n_inds = min(n_inds, max_inds)
        inds = np.arange(n_inds)

        # Handle n_iter setup differently for n_per = -1 vs normal case
        if n_per == -1:
            # For n_per = -1, we'll handle n_iter per chunk, not per (ind, ind_j) pair
            if isinstance(n_iter, int):
                n_iter_value = n_iter
            elif n_iter is None:
                n_iter_value = 5000
            else:
                # If it's an iterator, we'll seed per chunk for deterministic behavior
                n_iter_iter = n_iter
        else:
            # Original logic for n_per > 0
            if isinstance(n_iter, int):
                # make max_n an iter that always returns max_n
                n_iter = [n_iter] * n_inds * n_per
                # make an iterator
                n_iter = iter(n_iter)
            elif n_iter is None:
                n_iter = iter([5000] * n_inds * n_per)

        data = []
        for ind in inds:
            if max_total is not None and len(data) >= max_total:
                break

            if n_per == -1:
                # Generate ALL samples from this instance
                # For n_per = -1, we want to get as many samples as possible
                # Use a very large number or handle based on replacement setting
                if (
                    hasattr(self, "replacement")
                    and not self.replacement
                    and hasattr(self, "categories")
                ):
                    # For categorical without replacement, max is the number of categories
                    all_max_n = len(self.categories)
                else:
                    # For other cases or with replacement, use a large number
                    all_max_n = 100000  # Large number to get "all" samples

                # Set seed for deterministic sample generation
                np.random.seed(seed + ind)
                all_samples = self.get_raw_samples(
                    seed=seed + ind,
                    max_n=all_max_n,
                    ind=ind,
                )

                # Split samples into chunks based on n_iter
                chunk_start = 0
                chunk_ind = 0
                while chunk_start < len(all_samples):
                    if max_total is not None and len(data) >= max_total:
                        break

                    # Determine chunk size
                    if isinstance(n_iter, int) or n_iter is None:
                        chunk_size = n_iter_value
                    else:
                        # Set seed based on current state for deterministic chunk sizes
                        iter_seed = (
                            seed + ind + chunk_ind * 1000
                        )  # Unique seed for each chunk
                        np.random.seed(iter_seed)
                        chunk_size = next(n_iter_iter)

                    # Get chunk of samples
                    chunk_end = min(chunk_start + chunk_size, len(all_samples))
                    samples = all_samples[chunk_start:chunk_end]

                    # Process this chunk like the original logic
                    this_seed = seed + ind + chunk_ind * self.length
                    np.random.seed(this_seed)

                    for perm_ind in range(permutations_per_iter):
                        format_seed = this_seed + perm_ind
                        # set random seed
                        message_args = {
                            "seed": format_seed,
                            "permute_seed": abs(perm_ind - this_seed),
                            "include_description": np.random.rand()
                            < include_description_prob,
                        }

                        messages = self.samples_to_messages(
                            samples,
                            **message_args,
                        )

                        row = {
                            "name": self.name,
                            "dataset_ind": int(ind),
                            "repeat_ind": int(chunk_ind),
                            "seed": int(this_seed),
                            "include_description": message_args["include_description"],
                            "perm_ind": int(perm_ind),
                            "samples": samples,
                            "n": len(samples),
                            "messages": messages,
                            "max_n": len(samples),
                        }
                        data.append(row)

                    # Move to next chunk
                    chunk_start = chunk_end
                    chunk_ind += 1
            else:
                # Original logic for n_per > 0
                for ind_j in range(n_per):
                    this_seed = seed + ind + ind_j * self.length
                    np.random.seed(this_seed)
                    this_n = next(n_iter)

                    # get samples
                    samples = self.get_raw_samples(
                        seed=this_seed,
                        max_n=this_n,
                        ind=ind,
                    )
                    for perm_ind in range(permutations_per_iter):
                        format_seed = this_seed + perm_ind
                        # if not vary_formatter:
                        #     format_seed = this_seed
                        message_args = {
                            "seed": format_seed,
                            "permute_seed": abs(perm_ind - this_seed),
                            "include_description": np.random.rand()
                            < include_description_prob,
                            # "include_data_formatter_description": np.random.rand() < include_data_formatter_description_prob,
                            # "include_replacement": np.random.rand() < include_replacement_prob,
                            # "include_tags": np.random.rand() < include_tags_prob,
                        }

                        # if is_spectrum or is_colon or is_simple or is_bracket or is_special:
                        # if is_spectrum or is_colon or is_simple or is_bracket or is_special or is_chat:
                        # if True:
                        #     if tokenizer is None:
                        #         raise ValueError("tokenizer is required for spectrum formatters")
                        #     breakpoint()
                        #  texts, additional_info = self.samples_to_messages(
                        messages = self.samples_to_messages(
                            samples,
                            # tokenizer=tokenizer,
                            # loss_on_start_token=loss_on_start_token,
                            # format_type=format_type,
                            **message_args,
                        )

                        row = {
                            "name": self.name,
                            "dataset_ind": ind,
                            "include_description": message_args["include_description"],
                            "repeat_ind": ind_j,
                            "seed": this_seed,
                            "perm_ind": perm_ind,
                            "samples": samples,
                            "n": len(samples),
                            "messages": messages,
                            "max_n": this_n,
                        }
                        # # add additional params
                        # if self.additional_params:
                        #     row.update(self.additional_params)
                        # if additional_info:
                        #     if 'texts' in additional_info:
                        #         raise ValueError("texts is not allowed in additional_info")
                        #     row.update(additional_info)
                        data.append(row)
        return pd.DataFrame(data)


def format_sample(
    sample: Dict[str, Any],
    example_ind: int,
    gen_variables: List[str] | str,
    given_variables: List[str] | str | None = None,
) -> List[Dict[str, str]]:
    # make sure that the
    messages = []

    if len(given_variables) > 0:  # if there is an input
        # check if given_variables is a string
        if isinstance(given_variables, str):
            given_variables = [given_variables]
        given_dict = {given_var: sample[given_var] for given_var in given_variables}
        given_dict_json_str = json.dumps(given_dict)
        if len(given_variables) == 1:
            given_dict_json_str = str(given_dict[given_variables[0]])
        messages.append(
            {
                "role": "input",
                "content": given_dict_json_str,
                "example_ind": example_ind,
            }
        )
    # if gen variables is a string, make it a list
    if isinstance(gen_variables, str):
        gen_variables = [gen_variables]
    # check if sample has gen_variables
    gen_dict = {gen_var: sample[gen_var] for gen_var in gen_variables}
    gen_dict_json_str = json.dumps(gen_dict)
    if len(gen_variables) == 1:
        gen_dict_json_str = str(gen_dict[gen_variables[0]])
    messages.append(
        {"role": "output", "content": gen_dict_json_str, "example_ind": example_ind}
    )
    return messages


class MultivariateICLClass(ICLClass):
    def __init__(
        self,
        given_variables: List[str],
        gen_variables: List[str],
        name: str = "MultivariateRandomClass",
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        length: Optional[int] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, descriptions, length, additional_params)
        self.given_variables = given_variables
        self.gen_variables = gen_variables


def geom_iter(mean: float = 10):
    while True:
        yield np.random.geometric(1 / mean)


def poisson_iter(mean: float = 10):
    while True:
        yield np.random.poisson(mean)


def uniform_iter(low: float = 0, high: float = 1):
    while True:
        yield np.random.uniform(low, high)


def geom_and_poisson_iter(mean: float = 10):
    while True:
        yield np.random.geometric(1 / (mean / 2)) + np.random.poisson((mean / 2))
