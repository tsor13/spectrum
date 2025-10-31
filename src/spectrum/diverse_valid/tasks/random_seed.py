"""
uv run src/spectrum/diverse_valid/tasks/random_seed.py
"""

from spectrum.diverse_valid.generation_task import FunctionTask


# validation function is whether or not it's a number between 0 and 2**32 - 1
def is_valid_seed(x: str) -> bool:
    try:
        seed = int(x)
        return 0 <= seed < 2**32
    except ValueError:
        return False


def random_seed():
    return FunctionTask(
        name="random_seed",
        description="Generate a number to use for a random seed.",
        examples=[
            "15",
            "420",
            "8392013",
        ],
        validation_fn=is_valid_seed,
        max_new_tokens=4,
    )


def test_random_seed_validation():
    """Test random seed validation with comprehensive cases."""
    task = random_seed()

    # Valid seed cases (0 to 2^32-1)
    valid_cases = [
        "0",  # Minimum
        "42",  # Small number
        "1000000",  # Medium number
        "4294967295",  # Maximum (2^32-1)
        "123456789",  # Random valid
        " 42 ",  # With spaces
        "42\n",  # With newline
    ]

    # Invalid cases
    invalid_cases = [
        "-1",  # Negative
        "4294967296",  # Too large (2^32)
        "one",  # Text
        "1.5",  # Decimal
        "",  # Empty
        "invalid",  # Random text
        "0x42",  # Hex format
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Random Seed Task Test ===")
    task = random_seed()
    print(f"Task: {task.name}")
    print(f"Messages: {task.messages}")
    print()

    # Run validation tests
    print("Running validation tests...")
    test_random_seed_validation()
