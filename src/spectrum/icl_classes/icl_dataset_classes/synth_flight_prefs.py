import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectrum.icl_classes.individual_multivariate import IndividualMultivariate

""" Sample data instance with num_inc=2, num_flights=2, individual_description=False
<description>The following express flight preferences for the same individual among a set of flights. Predict which flight the indvidual prefers.</description>
Flights:
Flight 1:
Departure Time: 01:15 PM, Duration: 13 hr 50 min, Number of Stops: 1, Price: $610.00
Flight 2:
Departure Time: 06:14 PM, Duration: 3 hr 49 min, Number of Stops: 2, Price: $630.00

Rating: Flight 1 is best.

Flights:
Flight 1:
Departure Time: 07:00 PM, Duration: 12 hr 27 min, Number of Stops: 2, Price: $1530.00
Flight 2:
Departure Time: 07:58 PM, Duration: 8 hr 36 min, Number of Stops: 1, Price: $670.00

Rating: Flight 2 is best.
"""


def sample_profile(rng, num_entries=4):
    """
    Follows Appendix A.1. of https://arxiv.org/pdf/2503.17523v1

    This obtains the reward functions (profile).

    Assumes four components matter about flights
    """
    profile = np.zeros(num_entries)
    while (profile == np.zeros(num_entries)).all():
        profile = rng.integers(low=-2, high=2, size=num_entries) / 2
    return profile


def sample_flights(rng, num_flights):
    """Follows Appendix A.1. of https://arxiv.org/pdf/2503.17523v1

    This obtains candidate flights.
    """

    # assumes our profile has 4 components
    flights = np.zeros((num_flights, 4))
    flights[:, :3] = rng.choice(
        np.linspace(0, 1, 100), size=(num_flights, 3), replace=True
    )
    flights[:, 3] = rng.choice(np.linspace(0, 1, 3), size=(num_flights), replace=True)
    return flights


def get_flights_str(flights):
    flight_str = "\n"
    for fi, flight in enumerate(flights):

        # interpolate between 5am and midnight (flight[0])
        start = datetime(2025, 1, 1, 5, 0, 0)
        end = datetime(2025, 1, 2, 0, 0, 0)
        depart_time = datetime.strftime(start + (end - start) * flight[0], "%I:%M %p")

        # interpolate between 45 min and 16 hours
        min_dur, max_dur = datetime.strptime("00:45", "%H:%M"), datetime.strptime(
            "16:00", "%H:%M"
        )
        d = min_dur + (max_dur - min_dur) * flight[1]
        duration = f"{d.hour} hr {d.minute} min"

        min_price, max_price = 50, 1_500
        price = 10 * (min_price + (max_price - min_price) * flight[2] // 10)

        num_stops = int(flight[3] * 2)

        flight_str += f"Flight {fi + 1}:\nDeparture Time: {depart_time}, Duration: {duration}, Number of Stops: {num_stops}, Price: ${price:.2f}\n"
    return flight_str.strip()


def synth_flight_prefs_helper(
    seed=42, num_inc=3, num_flights=3, individual_description=False
):
    """
    :num_flights: number of flights per in-context example
    :inc: number of in-context examples
    :individual_description: whether to include the profile in the description
    """

    description = "The following express flight preferences for the same individual among a set of flights. Predict which flight the indvidual prefers."

    sample_df = {
        "individual_id": [],
        "Flights": [],
        "Rating": [],
        "individual_prior": [],
    }

    rng = np.random.default_rng(seed)
    profile = sample_profile(rng)
    for _ in range(num_inc):
        flights = sample_flights(rng, num_flights)
        best_flight = (flights @ profile).argmax()

        flights_str = get_flights_str(flights)
        # rating_str = f"Flight {best_flight + 1} is best."
        rating_str = f"{best_flight + 1}"
        # now turn these into strings
        sample_df["Flights"].append(flights_str)
        sample_df["Rating"].append(rating_str)
        sample_df["individual_id"].append("id")
        sample_df["individual_prior"].append(
            "The following express flight preferences for the same individual among a set of flights. "
            f"This person assigns a weight of {profile[0]} to flight departure time, {profile[1]} to flight duration, {profile[3]} to number of stops, and {profile[2]} to price. "
            "Predict which flight the indvidual prefers."
        )
    sample_df = pd.DataFrame(sample_df)

    return IndividualMultivariate(
        df=sample_df,
        individual_id_column="individual_id",
        individual_description_column=(
            "individual_prior" if individual_description else None
        ),
        given_variables=["Flights"],
        gen_variables=["Rating"],
        descriptions=[
            description
        ],  # This will be overwritten in individual_description is not None
        name="flight_prefs_individual",
    )


# TAYLOR - made these arguments less diverse as we're focusing on eval for this task
def generate_synth_flight_prefs(n_flights=3, individual_description=False, **kwargs):
    args = {
        "seed": 42,
        "n_iter": 20,  # number of icl examples in context
        "max_inds": 200,
    }
    args.update(kwargs)

    dfs = []
    for ind_i in range(args["max_inds"]):
        # increment seed
        args["seed"] += 1
        gen = synth_flight_prefs_helper(
            seed=args["seed"],
            num_flights=n_flights,
            num_inc=args["n_iter"],
            individual_description=individual_description,
        )
        data = gen.generate_many(**args)
        data["seed"] = args["seed"]
        data["num_flights"] = n_flights
        data["num_ic_examples"] = args["n_iter"]
        dfs.append(data)
    return pd.concat(dfs)


np.random.seed(42)
