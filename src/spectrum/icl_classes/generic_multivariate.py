import glob
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.icl_class import MultivariateICLClass, format_sample


class GenericMultivariate(MultivariateICLClass):
    def __init__(
        self,
        df: pd.DataFrame,
        given_variables: List[str],
        gen_variables: List[str],
        name: str,
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        do_shuffle: bool = True,
        group_by: Optional[List[str]] = None,
    ):
        """
        Bernoulli random class.
        :param name: Identifier.
        :param p: Probability of success; if None, sampled randomly.
        :param n_samples: Number of samples; if None, sampled randomly.
        :param names: Labels for outcomes; if None, chosen randomly.
        :param seed: Base random seed.
        """
        self.df = df
        # assert that variables are in df
        for var in given_variables:
            assert var in df.columns, f"{var} is not in df"
        for var in gen_variables:
            assert var in df.columns, f"{var} is not in df"
        if name is None:
            raise ValueError("Name is required for GenericMultivariate class.")

        if descriptions is None:
            descriptions = []
            given_variables_str = ", ".join(given_variables)
            gen_variables_str = ", ".join(gen_variables)
            descriptions.append(
                f"The following are given: {given_variables_str}. You must generate {gen_variables_str}."
            )
            descriptions.append(
                f"Given: {given_variables_str}. Generate: {gen_variables_str}."
            )
            descriptions.append(
                f"Inputs: {given_variables_str}. Outputs: {gen_variables_str}."
            )

        self.do_shuffle = do_shuffle

        self.group_by = group_by

        super().__init__(
            name=name,
            given_variables=given_variables,
            gen_variables=gen_variables,
            descriptions=descriptions,
            additional_params=additional_params,
            length=1,
        )

    def get_raw_samples(
        self,
        seed: int,
        max_n: int,
        ind: int | None = None,
    ) -> List[Dict[str, Any]]:
        self.verify_ind(ind)
        include_vars = self.given_variables + self.gen_variables
        if self.group_by is not None:
            include_vars += self.group_by
        df = self.df.copy()[include_vars]
        # shuffle
        if self.do_shuffle:

            # samples = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            if self.group_by is not None:
                # Shuffle the groups, then concatenate the groups in their new order,
                # preserving the within-group ordering (but if you want to shuffle order within group, also sample within each group)
                grouped = [g for _, g in df.groupby(self.group_by)]
                rng = np.random.RandomState(seed)
                rng.shuffle(grouped)
                samples = pd.concat(grouped, ignore_index=True)
                # reset index
                samples = samples.reset_index(drop=True)
            else:
                samples = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            samples = df.reset_index(drop=True)
        # get first max_n
        samples = samples.head(max_n)
        # to list of dicts
        samples = samples.to_dict(orient="records")
        return samples

    def sample_to_messages(
        self,
        sample: Dict[str, Any],
        example_ind: int,
    ) -> List[Dict[str, Any]]:
        messages = format_sample(
            sample=sample,
            example_ind=example_ind,
            gen_variables=self.gen_variables,
            given_variables=self.given_variables,
        )
        return messages
