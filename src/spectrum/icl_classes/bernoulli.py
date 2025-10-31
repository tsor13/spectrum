from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.icl_class import ICLClass


class Bernoulli(ICLClass):
    def __init__(
        self,
        p: Optional[float] = None,
        name: Optional[str] = None,
        sample_names: List[str] = None,
        descriptions: Optional[Union[List[str], Dict[str, str]]] = None,
        # sample_space: Optional[Union[str, List[str]]] = None,
        # data_formatter: Optional[List[DataFormatter]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Bernoulli random class.
        :param name: Identifier.
        :param p: Probability of success; if None, sampled randomly.
        :param n_samples: Number of samples; if None, sampled randomly.
        :param names: Labels for outcomes; if None, chosen randomly.
        :param seed: Base random seed.
        """
        # determine parameters
        self.p = p if p is not None else random_p(seed)
        if name is None:
            name = f"Bernoulli(p={round(self.p, 3)})"

        if random_seed is not None:
            np.random.seed(random_seed)

        if sample_names is None:
            # raise Exception("Need to include sample names for draws")
            # randomly choose between
            possible_names = [
                ["Heads", "Tails"],
                ["heads", "tails"],
                ["h", "t"],
                ["Success", "Failure"],
                ["A", "B"],
                ["1", "0"],
                ["Win", "Lose"],
                ["Yes", "No"],
                ["True", "False"],
                ["T", "F"],
            ]
            sample_names = possible_names[np.random.randint(0, len(possible_names))]
        if len(sample_names) != 2:
            raise Exception("Need to include 2 sample names for draws")
        self.sample_names = sample_names
        if descriptions is None:
            descriptions = []
            # descriptions.append(f"Bernoulli(p={round(self.p, 3)}), Options: {self.sample_names}")
            descriptions.append(str(self.sample_names))
            # add a dictionary with probabilities
            descriptions.append(
                str(
                    {
                        k: float(round(v, 3))
                        for k, v in zip(self.sample_names, [self.p, 1 - self.p])
                    }
                )
            )
            descriptions.append(
                f"Bernoulli({str({k: float(round(v, 3)) for k, v in zip(self.sample_names, [self.p, 1-self.p])})})"
            )
            descriptions.append(
                str(
                    {
                        k: f"{float(100*round(v, 3))}%"
                        for k, v in zip(self.sample_names, [self.p, 1 - self.p])
                    }
                )
            )
            descriptions.append(
                f"Bernoulli({str({k: f'{float(100*round(v, 3))}%' for k, v in zip(self.sample_names, [self.p, 1-self.p])})})"
            )
            # just say which is more likely than the other
            if self.p > 0.5:
                descriptions.append(
                    f"Options: {self.sample_names}\n{self.sample_names[0]} is more likely than {self.sample_names[1]}."
                )
            elif self.p < 0.5:
                descriptions.append(
                    f"Options: {self.sample_names}\n{self.sample_names[1]} is more likely than {self.sample_names[0]}."
                )
            else:
                descriptions.append(
                    f"Options: {self.sample_names}\n{self.sample_names[0]} and {self.sample_names[1]} are equally likely."
                )
            descriptions.append(
                f"Coin flip with {self.sample_names[0]} and {self.sample_names[1]}. It is potentially biased."
            )
        if additional_params is None:
            additional_params = {}
        additional_params["p"] = p

        super().__init__(
            name=name,
            descriptions=descriptions,
            # sample_space=sample_space,
            # data_formatter=data_formatter,
            additional_params=additional_params,
        )

    def generate_data(self, n_samples: int, seed: int) -> pd.DataFrame:
        np.random.seed(seed + 10)
        # return np.random.normal(mean, std, (n_samples, round))
        gens = np.random.binomial(1, 1 - self.p, (n_samples))
        return gens

    def get_raw_samples(
        self,
        seed: int,
        max_n: int,
        ind: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate coin flip texts.
        """
        # if ind is not None:
        #     raise NotImplementedError("Indexing not implemented for Bernoulli class.")
        self.verify_ind(ind)
        draws = self.generate_data(max_n, seed * 2)
        # to dictionary
        samples = [{"draw": int(draw), "data_id": i} for i, draw in enumerate(draws)]
        return samples

    def sample_to_messages(
        self,
        sample: Dict[str, Any],
        example_ind: int,
        # sample_space: Optional[List[str]] = None,
        # data_formatter: Optional[DataFormatter] = None,
    ) -> List[Dict[str, Any]]:
        draw = sample["draw"]
        draw_text = self.sample_names[draw]
        return [{"role": "output", "content": draw_text, "example_ind": example_ind}]
