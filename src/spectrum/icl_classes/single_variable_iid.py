from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.icl_class import ICLClass, geom_and_poisson_iter, uniform_iter


class SingleVariableIID(ICLClass):
    def __init__(
        self,
        categories: List[str],
        replacement: bool,
        category_probs: Optional[List[float]] = None,
        name: Optional[str] = None,
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        # sample_space: Optional[Union[str, List[str]]] = None,
        # data_formatter: Optional[List[DataFormatter]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        max_n_include_description: int = 10,
    ):
        # super().__init__(name, descriptions, None, data_formatter, additional_params)
        super().__init__(name, descriptions, additional_params, replacement=replacement)
        self.categories = categories
        if category_probs is None:
            category_probs = [1 / len(categories)] * len(categories)
        category_probs = np.array(category_probs)
        category_probs = category_probs / category_probs.sum()
        self.category_probs = category_probs
        self.cat_to_prob = {
            str(cat): float(prob) for cat, prob in zip(categories, category_probs)
        }
        if descriptions is None:
            replace_strs = [""]
            if not replacement:
                replace_strs = [
                    "Replacement: False",
                    "No replacement",
                    "No repeats",
                    "No duplicates",
                    "Without replacement",
                ]
            else:
                replace_strs = [
                    "Replacement: True",
                    "Duplicates allowed",
                    "w/ repeats",
                    "With replacement",
                ]
            dist_descriptions = [""]
            if len(categories) < max_n_include_description:
                # print the probabilities as description
                dist_descriptions.append(str(self.cat_to_prob))
                dist_descriptions.append(f"Options: {', '.join(categories)}")
                dist_descriptions.append(
                    "\n".join(
                        [f"{cat}: {prob}" for cat, prob in self.cat_to_prob.items()]
                    )
                )
            descriptions = []
            for replace_str in replace_strs:
                for dist_description in dist_descriptions:
                    descriptions.append(f"{replace_str}\n{dist_description}")
                    descriptions.append(f"{dist_description}\n{replace_str}")
            self.descriptions = descriptions

        self.max_n_include_description = max_n_include_description

    def get_raw_samples(
        self, seed: int, max_n: int, ind: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not self.replacement:
            max_n = min(max_n, len(self.categories))
        inds = np.random.choice(
            len(self.categories),
            size=max_n,
            p=self.category_probs,
            replace=self.replacement,
        )
        return [{"sample": self.categories[ind]} for ind in inds]

    def sample_to_messages(
        self, sample: Dict[str, Any], example_ind: int
    ) -> List[Dict[str, Any]]:
        return [
            {"role": "output", "content": sample["sample"], "example_ind": example_ind}
        ]
