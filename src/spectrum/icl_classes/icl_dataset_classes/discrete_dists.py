import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.generic_multivariate import GenericMultivariate
from spectrum.icl_classes.icl_class import geom_and_poisson_iter
from spectrum.icl_classes.individual_multivariate import IndividualMultivariate
from spectrum.icl_classes.single_variable_iid import SingleVariableIID


def generate_cards(**kwargs):
    """
    Generate a dataset of card draws from a 52-card deck without replacement.
    Each card is treated as a categorical class.
    """
    args = {
        "seed": 42,
        "n_per": 100,
        # "max_inds": 1,
    }
    args.update(kwargs)

    # Set random seed
    np.random.seed(args["seed"])

    # Create the deck of cards
    # suits = ['hearts', 'diamonds', 'clubs', 'spades']
    # clubs (♣), diamonds (♦), hearts (♥), and spades (♠).
    suits = ["♣", "♦", "♥", "♠"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    deck = [f"{rank}{suit}" for suit in suits for rank in ranks]

    return SingleVariableIID(
        deck,
        name="cards",
        descriptions=[
            "Draw cards at random from a standard 52-card deck without replacement. The deck contains all standard playing cards (2-10, J, Q, K, A) in four suits (♣, ♦, ♥, ♠)."
        ],
        replacement=False,
    ).generate_many(**args)


def generate_poisson(
    description_gamma_shape=0.5,
    description_gamma_scale=2000,
    num_draws_iter=geom_and_poisson_iter(mean=1024),
    **kwargs,
):
    """
    Generate a dataset of poisson draws.
    """
    args = {
        "seed": 42,
        # "n_iter": geom_and_poisson_iter(mean=1024),
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    # almost uninformative description, to have most values fall between 0 and 1000 is gamma(1, 1)

    dfs = []
    for i in range(args["max_inds"]):
        lambda_ = np.random.gamma(description_gamma_shape, description_gamma_scale)
        # draw a poisson draw
        # draws = np.random.poisson(lambda_, 10_000)
        draws = np.random.poisson(lambda_, next(num_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        np.random.seed(args["seed"] + i)
        # categorical add
        df = SingleVariableIID(
            draws,
            name=f"poisson",
            descriptions=[
                f"Draws from a Poisson distribution with parameter {lambda_}.",
                f"Poisson distribution",
                f"Poisson({lambda_})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["lambda"] = lambda_
        dfs.append(df)
    return pd.concat(dfs)


def generate_geometric(num_draws_iter=geom_and_poisson_iter(mean=1024), **kwargs):
    """
    Generate a dataset of geometric draws.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample p from beta(1, 1)
        p = np.random.beta(1, 1)
        # draw a geometric draw
        draws = np.random.geometric(p, next(num_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        df = SingleVariableIID(
            draws,
            name=f"geometric",
            descriptions=[
                f"Draws from a geometric distribution with parameter {p}.",
                f"Geometric distribution",
                f"Geometric({p})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["p"] = p
        dfs.append(df)
    return pd.concat(dfs)


def generate_bernoulli(num_draws_iter=geom_and_poisson_iter(mean=1024), **kwargs):
    """
    Generate a dataset of bernoulli draws.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample p from beta(1, 1)
        p = np.random.beta(1, 1)
        # draw bernoulli draws
        draws = np.random.binomial(1, p, next(num_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"bernoulli",
            descriptions=[
                f"Draws from a bernoulli distribution with parameter {p}.",
                f"Bernoulli distribution",
                f"Bernoulli({p})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["p"] = p
        dfs.append(df)
    return pd.concat(dfs)


def generate_binomial(num_draws_iter=geom_and_poisson_iter(mean=1024), **kwargs):
    """
    Generate a dataset of binomial draws.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample p from beta(1, 1)
        p = np.random.beta(1, 1)
        # sample n from geometric distribution
        n = next(num_draws_iter)
        # draw binomial draws
        draws = np.random.binomial(n, p, next(num_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"binomial",
            descriptions=[
                f"Draws from a binomial distribution with parameters n={n} and p={p}.",
                f"Binomial distribution",
                f"Binomial({n}, {p})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["p"] = p
        df["n_trials"] = n
        dfs.append(df)
    return pd.concat(dfs)


def generate_negative_binomial(
    num_draws_iter=geom_and_poisson_iter(mean=1024), **kwargs
):
    """
    Generate a dataset of negative binomial draws.
    The negative binomial distribution represents the number of failures before r successes
    in a sequence of independent Bernoulli trials with probability p of success.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample r from beta(1, 1)
        r = np.random.beta(1, 1)
        # sample p from uniform(0, 1)
        p = np.random.uniform(0, 1)
        # draw negative binomial draws
        draws = np.random.negative_binomial(r, p, next(num_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"negative_binomial",
            descriptions=[
                f"Draws from a negative binomial distribution with parameters r={r:.2f} and p={p:.2f}.",
                f"Negative binomial distribution",
                f"NegativeBinomial({r:.2f}, {p:.2f})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["r"] = r
        df["p"] = p
        dfs.append(df)
    return pd.concat(dfs)


def generate_categorical(
    category_n_iter=geom_and_poisson_iter(12),
    n_draws_iter=geom_and_poisson_iter(1024),
    **kwargs,
):
    """
    Generate a dataset of categorical draws.
    The categorical distribution represents draws from a discrete probability distribution
    over a finite set of categories.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample number of categories
        n_categories = next(category_n_iter)
        # make sure n_categories is at least 2
        n_categories = max(n_categories, 2)
        # sample probability vector from Dirichlet(1,1,...,1)
        p = np.random.dirichlet(np.ones(n_categories))
        # draw categorical draws
        draws = np.random.choice(n_categories, size=next(n_draws_iter), p=p)
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        prob_vector_string = str({i: float(np.round(p_, 4)) for i, p_ in enumerate(p)})
        df = SingleVariableIID(
            draws,
            name=f"categorical",
            descriptions=[
                f"Draws from a categorical distribution with {n_categories} categories and probability vector {prob_vector_string}.",
                f"Categorical distribution",
                f"SingleVariableIID({n_categories}, {prob_vector_string})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["n_categories"] = n_categories
        # df['p'] = p
        dfs.append(df)
    return pd.concat(dfs)


def generate_multinomial(
    category_n_iter=geom_and_poisson_iter(12),
    multinomial_n_iter=geom_and_poisson_iter(32),
    n_draws_iter=geom_and_poisson_iter(1024),
    **kwargs,
):
    """
    Generate a dataset of multinomial draws.
    The multinomial distribution represents draws from a discrete probability distribution
    over a finite set of categories, where each draw can result in multiple counts per category.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample number of categories
        n_categories = next(category_n_iter)
        # make sure n_categories is at least 2
        n_categories = max(n_categories, 2)
        # sample probability vector from Dirichlet(1,1,...,1)
        p = np.random.dirichlet(np.ones(n_categories))
        # draw multinomial draws
        n_draws = next(n_draws_iter)
        multinomial_n = next(multinomial_n_iter)
        draws = np.random.multinomial(multinomial_n, p, n_draws)
        # to strings
        # randomize whether as dict or list
        if np.random.rand() < 0.5:
            draws = [str({i: d for i, d in enumerate(draw)}) for draw in draws]
        else:
            draws = [str([str(d) for d in draw]) for draw in draws]
        # increment seed
        args["seed"] += 1
        prob_vector_string = str({i: float(np.round(p_, 4)) for i, p_ in enumerate(p)})
        df = SingleVariableIID(
            draws,
            name=f"multinomial",
            descriptions=[
                f"Draws from a multinomial distribution with {n_categories} categories, {multinomial_n} total draws, and probability vector {prob_vector_string}.",
                f"Multinomial distribution",
                f"Multinomial({n_categories}, {multinomial_n}, {prob_vector_string})",
            ],
            replacement=False,
        ).generate_many(**args)
        df["n_categories"] = n_categories
        df["n_draws"] = n_draws
        dfs.append(df)
    return pd.concat(dfs)


def generate_hypergeometric(n_draws_iter=geom_and_poisson_iter(1024), **kwargs):
    """
    Generate a dataset of hypergeometric draws.
    The hypergeometric distribution represents the probability of k successes in n draws
    without replacement from a finite population of size N containing exactly K successes.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample population size N
        N = next(geom_and_poisson_iter(128))
        # sample number of successes K in population
        K = np.random.randint(1, N + 1)
        # sample number of draws n
        n = np.random.randint(1, N + 1)
        # draw hypergeometric draws
        draws = np.random.hypergeometric(K, N - K, n, next(n_draws_iter))
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"hypergeometric",
            descriptions=[
                f"Draws from a hypergeometric distribution with population size N={N}, number of positive draws K={K}, number of draws n={n}.",
                f"Hypergeometric distribution",
            ],
            replacement=False,
        ).generate_many(**args)
        df["N"] = N
        df["K"] = K
        df["n_draws"] = n
        dfs.append(df)
    return pd.concat(dfs)


def generate_geometric_beta(n_draws_iter=geom_and_poisson_iter(1024), **kwargs):
    """
    Generate a dataset of geometric draws where the success probability p
    is drawn from a Beta(1,1) distribution (uniform on [0,1]).
    The geometric distribution represents the number of failures before the first success
    in a sequence of independent Bernoulli trials with success probability p.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        # sample success probability p from Beta(1,1)
        p = np.random.beta(1, 1)
        # sample number of draws
        n_draws = next(n_draws_iter)
        # draw geometric draws
        draws = np.random.geometric(p, n_draws)
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"geometric_beta",
            descriptions=[
                f"Draws from a geometric distribution with success probability p={p:.3f}.",
                f"Geometric distribution",
            ],
            replacement=False,
        ).generate_many(**args)
        df["p"] = p
        df["n_draws"] = n_draws
        dfs.append(df)
    return pd.concat(dfs)


def generate_zipfian(n_draws_iter=geom_and_poisson_iter(1024), **kwargs):
    """
    Generate a dataset of Zipfian draws where the alpha parameter
    is drawn from Geometric(1/4)+.00001 distribution.
    The Zipfian distribution is a discrete power law distribution.
    """
    args = {
        "seed": 42,
        "max_inds": 500,
    }
    args.update(kwargs)
    # set seed
    np.random.seed(args["seed"])

    dfs = []
    for i in range(args["max_inds"]):
        alpha = np.random.uniform(1 + 1e-6, 2.5)
        # sample number of draws
        n_draws = next(n_draws_iter)
        # draw zipfian draws
        draws = np.random.zipf(alpha, n_draws)
        # to strings
        draws = [str(draw) for draw in draws]
        # increment seed
        args["seed"] += 1
        df = SingleVariableIID(
            draws,
            name=f"zipfian",
            descriptions=[
                f"Draws from a Zipfian distribution with alpha={alpha:.3f}.",
                f"Zipfian distribution",
            ],
            replacement=False,
        ).generate_many(**args)
        df["alpha"] = alpha
        df["n_draws"] = n_draws
        dfs.append(df)
    return pd.concat(dfs)
