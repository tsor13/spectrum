"""
Weekday generation tasks.

uv run python src/spectrum/diverse_valid/tasks/weekdays.py
"""

from spectrum.diverse_valid.generation_task import InclusionTask

# Full weekday names
WEEKDAYS_FULL = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

# 3-letter abbreviations
WEEKDAYS_ABBREVIATED = [
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
    "Mon.",
    "Tue.",
    "Wed.",
    "Thu.",
    "Fri.",
    "Sat.",
    "Sun.",
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
    "SUN",
]

# Single letter abbreviations (sometimes used)
WEEKDAYS_SINGLE_LETTER = [
    "M",
    "T",
    "W",
    "Th",
    "F",
    "Sa",
    "Su",  # R for Thursday, U for Sunday to avoid conflicts
]


def weekdays_full():
    """Weekdays - full names only."""
    return InclusionTask(
        name="weekdays_full",
        description="Name a day of the week",
        examples=["Thursday", "Wednesday", "Sunday"],
        valid_strings=WEEKDAYS_FULL,
        max_new_tokens=10,
    )


def weekdays_abbreviated():
    """Weekdays - 3-letter abbreviations."""
    return InclusionTask(
        name="weekdays_abbreviated",
        description="Day of the week abbreviation",
        examples=["Thu", "Wed.", "SUN"],
        valid_strings=WEEKDAYS_ABBREVIATED,
        max_new_tokens=5,
    )


def weekdays_any_format():
    """Weekdays - full names or abbreviations."""
    return InclusionTask(
        name="weekdays_any_format",
        description="Day of the week (full name or abbreviation)",
        examples=["Monday", "Tue", "SUN"],
        valid_strings=WEEKDAYS_FULL + WEEKDAYS_ABBREVIATED + WEEKDAYS_SINGLE_LETTER,
        max_new_tokens=10,
    )


def test_weekdays_full_validation():
    """Test weekdays full names validation."""
    task = weekdays_full()

    # Valid full weekday names
    valid_cases = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
        "Mon.",
        "Tue.",
        "Wed.",
        "Thu.",
        "Fri.",
        "Sat.",
        "Sun.",
        "MON",
        "TUE",
        "WED",
        "THU",
        "FRI",
        "SAT",
        "SUN",
    ]

    # Invalid cases
    invalid_cases = [
        "Mon",  # Abbreviation (not allowed in full names task)
        "Tue",  # Abbreviation
        "monday",  # Wrong case
        "MONDAY",  # Wrong case
        "Mondayy",  # Misspelling
        "Weekday",  # Generic term
        "Today",  # Relative term
        "Holiday",  # Not a weekday
        "",  # Empty
        "8th day",  # Invalid
        "First day",  # Invalid
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_weekdays_abbreviated_validation():
    """Test weekdays abbreviations validation."""
    task = weekdays_abbreviated()

    # Valid weekday abbreviations
    valid_cases = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Invalid cases
    invalid_cases = [
        "Monday",  # Full name (not allowed in abbreviations task)
        "Tuesday",  # Full name
        "mon",  # Wrong case
        "MON",  # Wrong case
        "Mo",  # Too short
        "Mond",  # Too long
        "M",  # Single letter
        "T",  # Single letter (ambiguous)
        "",  # Empty
        "123",  # Numbers
        "ABC",  # Random letters
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_weekdays_any_format_validation():
    """Test weekdays any format validation."""
    task = weekdays_any_format()

    # Valid cases (both formats)
    valid_cases = [
        "Monday",
        "Mon",
        "Tuesday",
        "Tue",
        "Wednesday",
        "Wed",
        "Thursday",
        "Thu",
        "Friday",
        "Fri",
        "Saturday",
        "Sat",
        "Sunday",
        "Sun",
    ]

    # Invalid cases
    invalid_cases = [
        "monday",  # Wrong case for full name
        "mon",  # Wrong case for abbreviation
        "M",  # Single letter
        "Weekday",  # Generic term
        "8th day",  # Invalid
        "",  # Empty
        "Holiday",  # Not a weekday
        "Mondayy",  # Misspelling
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Weekdays Tasks Test ===")

    # Test full names
    print("\n--- Testing weekdays_full ---")
    task1 = weekdays_full()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_weekdays_full_validation()

    # Test abbreviations
    print("\n--- Testing weekdays_abbreviated ---")
    task2 = weekdays_abbreviated()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_weekdays_abbreviated_validation()

    # Test any format
    print("\n--- Testing weekdays_any_format ---")
    task3 = weekdays_any_format()
    print(f"Task: {task3.name}")
    print(f"Messages: {task3.messages}")
    test_weekdays_any_format_validation()
