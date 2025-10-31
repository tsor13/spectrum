"""
uv run random_classes/dataset_loaders/normal.py
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.icl_class import ICLClass, geom_and_poisson_iter


def random_mean_std(seed: int = 42) -> tuple[float, float]:
    np.random.seed(seed)
    p = np.random.rand()
    if p < 0.2:
        mean = np.random.uniform(0, 100)
        std = np.random.uniform(1, 5)
    elif p < 0.4:
        mean = np.random.uniform(0, 100)
        std = np.random.uniform(1, 5)
        mean *= np.exp(np.random.normal(0, 4))
        std *= np.exp(np.random.normal(0, 3))
    elif p < 0.6:
        mean = np.random.poisson(10)
        std = np.random.poisson(20)
        if np.random.rand() < 0.5:
            mean = -mean
        if np.random.rand() < 0.5:
            std = 1.0 / std if std != 0 else std
    else:
        mean = np.random.uniform(0, 100)
        std = np.random.uniform(1, 5)
    # randomly sample a number from 0 to 10, and exponentially scale
    scale = np.random.uniform(-10, 10)
    mean *= np.exp(scale)
    std *= np.exp(scale)
    return float(mean), float(std)


def random_round_digits(seed: int = 42) -> int:
    return int(np.random.poisson(3) + 1)


class Normal(ICLClass):
    def __init__(
        self,
        mean: float,
        std: float,
        round_digits: int = 3,
        name: Optional[str] = None,
        descriptions: Optional[List[str]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        if name is None:
            name = f"Normal(mean={mean}, std={std}, round_digits={round_digits})"
        if descriptions is None:
            param_strs = [
                (f"{mean:.5f}", f"{std:.5f}"),
                (f"{mean:.4f}", f"{std:.4f}"),
                (f"{mean:.3f}", f"{std:.3f}"),
                (f"{mean:.2f}", f"{std:.2f}"),
                (f"{mean:.4e}", f"{std:.4e}"),
                (f"{mean:.2e}", f"{std:.2e}"),
            ]
            start_strs = []
            for mean_str, std_str in param_strs:
                start_strs.append(
                    f"Normal distribution with mean {mean_str} and std {std_str}."
                )
                start_strs.append(
                    f"The following are draws from a normal distribution with mean {mean_str} and standard deviation {std_str}."
                )
                start_strs.append(f"x~N({mean_str}, {std_str})")
                start_strs.append(f"Normal distribution")
                start_strs.append(f"Floating point numbers")
                start_strs.append(f"Decimal numbers")
                start_strs.append(f"Numbers")

            round_digits_strs = [
                f"Precision: {round_digits}",
                f"{round_digits} decimal places",
                "",
            ]
            descriptions = []
            for start_str in start_strs:
                for round_digits_str in round_digits_strs:
                    descriptions.append(f"{start_str}\n{round_digits_str}")

        if additional_params is None:
            additional_params = {}
        additional_params.update(
            {"mean": mean, "std": std, "round_digits": round_digits}
        )

        super().__init__(
            name=name,
            descriptions=descriptions,
            additional_params=additional_params,
        )
        self.mean = mean
        self.std = std
        self.round_digits = round_digits

    def generate_data(self, n_samples: int, seed: int) -> List[float]:
        np.random.seed(seed + 10)
        samples = np.random.normal(self.mean, self.std, n_samples)
        return samples.tolist()

    def get_raw_samples(
        self, seed: int, max_n: int, ind: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        self.verify_ind(ind)
        draws = self.generate_data(max_n, seed * 2)
        return [{"draw": draw, "data_id": i} for i, draw in enumerate(draws)]

    def sample_to_messages(
        self,
        sample: Dict[str, Any],
        example_ind: int,
        # sample_space: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        draw_value = sample["draw"]
        draw_text = f"{draw_value:.{self.round_digits}f}"
        return [{"role": "output", "content": draw_text, "example_ind": example_ind}]


def generate_normal_data(n_normals: int = 1000, **kwargs) -> pd.DataFrame:
    """
    Generate DataFrame using draws from the Normal class.
    """
    args = {
        "seed": 42,
        "n_per": 1,
        "n_iter": geom_and_poisson_iter(mean=128),
        "include_description_prob": 0.4,
    }
    args.update(kwargs)

    all_data = []
    for i in tqdm(range(n_normals)):
        seed = args["seed"] + i
        mean, std = random_mean_std(seed)
        round_digits = random_round_digits(seed)
        normal = Normal(mean=mean, std=std, round_digits=round_digits)
        args["seed"] = seed
        data = normal.generate_many(**args)
        all_data.append(data)
    return pd.concat(all_data)
