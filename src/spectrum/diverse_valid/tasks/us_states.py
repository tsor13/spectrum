"""
US States generation tasks.

uv run python src/spectrum/diverse_valid/tasks/us_states.py
"""

from spectrum.diverse_valid.generation_task import InclusionTask

# Complete list of US states (full names)
US_STATES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

# US state abbreviations (2-letter codes)
US_STATE_ABBREVIATIONS = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def us_states_full_names():
    """US states - full names only."""
    return InclusionTask(
        name="us_states_full_names",
        description="Name a US state",
        examples=["Kentucky", "Utah", "Oregon"],
        valid_strings=US_STATES,
        max_new_tokens=15,
    )


def us_states_abbreviations():
    """US states - 2-letter abbreviations only."""
    return InclusionTask(
        name="us_states_abbreviations",
        description="US state abbreviation",
        examples=["KY", "UT", "OR"],
        valid_strings=US_STATE_ABBREVIATIONS,
        max_new_tokens=5,
    )


def us_states_any_format():
    """US states - either full names or abbreviations."""
    return InclusionTask(
        name="us_states_any_format",
        description="US state name or abbreviation",
        examples=["Kentucky", "UT", "Oregon"],
        valid_strings=US_STATES + US_STATE_ABBREVIATIONS,
        max_new_tokens=15,
    )


def test_us_states_full_names_validation():
    """Test US states full names validation."""
    task = us_states_full_names()

    # Valid full state names
    valid_cases = [
        "California",
        "Texas",
        "Florida",
        "New York",
        "Pennsylvania",
        "Illinois",
        "Ohio",
        "Georgia",
        "North Carolina",
        "Michigan",
    ]

    # Invalid cases
    invalid_cases = [
        "CA",  # Abbreviation (not allowed in full names task)
        "NY",  # Abbreviation
        "Los Angeles",  # City, not state
        "Washington DC",  # District, not state
        "Puerto Rico",  # Territory, not state
        "Canada",  # Country
        "InvalidState",  # Made up
        "",  # Empty
        "california",  # Wrong case (states should be title case)
        "CALIFORNIA",  # Wrong case
        "New york",  # Wrong case
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_us_states_abbreviations_validation():
    """Test US states abbreviations validation."""
    task = us_states_abbreviations()

    # Valid state abbreviations
    valid_cases = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]

    # Invalid cases
    invalid_cases = [
        "California",  # Full name (not allowed in abbreviations task)
        "Texas",  # Full name
        "DC",  # District, not state
        "PR",  # Territory, not state
        "XX",  # Invalid abbreviation
        "ZZ",  # Invalid abbreviation
        "",  # Empty
        "ca",  # Wrong case (should be uppercase)
        "C",  # Too short
        "CAL",  # Too long
        "123",  # Numbers
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_us_states_any_format_validation():
    """Test US states any format validation."""
    task = us_states_any_format()

    # Valid cases (both formats)
    valid_cases = [
        "California",
        "CA",
        "Texas",
        "TX",
        "New York",
        "NY",
        "Florida",
        "FL",
        "Illinois",
        "IL",
    ]

    # Invalid cases
    invalid_cases = [
        "Los Angeles",  # City
        "Washington DC",  # District
        "Puerto Rico",  # Territory
        "XX",  # Invalid abbreviation
        "InvalidState",  # Made up
        "",  # Empty
        "california",  # Wrong case for full name
        "ca",  # Wrong case for abbreviation
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== US States Tasks Test ===")

    # Test full names
    print("\n--- Testing us_states_full_names ---")
    task1 = us_states_full_names()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_us_states_full_names_validation()

    # Test abbreviations
    print("\n--- Testing us_states_abbreviations ---")
    task2 = us_states_abbreviations()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_us_states_abbreviations_validation()

    # Test any format
    print("\n--- Testing us_states_any_format ---")
    task3 = us_states_any_format()
    print(f"Task: {task3.name}")
    print(f"Messages: {task3.messages}")
    test_us_states_any_format_validation()
