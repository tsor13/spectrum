"""
Prime number generation tasks.

uv run python src/spectrum/diverse_valid/tasks/prime_numbers.py
"""

from spectrum.diverse_valid.generation_task import FunctionTask


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def validate_prime_number(text: str) -> bool:
    """Validate that text is a prime number."""
    # Don't allow leading/trailing spaces or leading zeros
    if text != text.strip():
        return False
    if text.startswith("0") and len(text) > 1:
        return False

    try:
        num = int(text)
        return is_prime(num)
    except ValueError:
        return False


def validate_small_prime(text: str) -> bool:
    """Validate that text is a prime number under 100."""
    if not validate_prime_number(text):
        return False

    try:
        num = int(text.strip())
        return num < 100
    except ValueError:
        return False


def prime_numbers():
    """Prime numbers (any size)."""
    return FunctionTask(
        name="prime_numbers",
        description="Generate a prime number",
        examples=[
            "617",
            "13",
            "47",
        ],
        validation_fn=validate_prime_number,
        max_new_tokens=10,
    )


def small_prime_numbers():
    """Prime numbers under 100."""
    return FunctionTask(
        name="small_prime_numbers",
        description="Generate a prime number less than 100",
        examples=[
            "29",
            "5",
            "97",
        ],
        validation_fn=validate_small_prime,
        max_new_tokens=5,
    )


def test_prime_numbers_validation():
    """Test prime numbers validation."""
    task = prime_numbers()

    # Valid prime numbers
    valid_cases = [
        "2",  # Smallest prime
        "3",
        "5",
        "7",
        "11",
        "13",
        "17",
        "19",
        "23",
        "29",  # Small primes
        "31",
        "37",
        "41",
        "43",
        "47",  # More small primes
        "97",
        "101",
        "103",
        "107",
        "109",  # Primes around 100
        "997",
        "1009",
        "1013",  # Larger primes
    ]

    # Invalid cases
    invalid_cases = [
        "1",  # Not prime by definition
        "0",  # Not prime
        "-1",  # Negative
        "4",  # Even composite
        "6",
        "8",
        "9",
        "10",
        "12",  # Small composites
        "15",
        "21",
        "25",
        "27",  # Larger composites
        "100",
        "121",
        "144",  # Perfect squares
        "1000",  # Large composite
        " 7 ",  # With spaces
        "07",  # Leading zero
        "3.5",  # Decimal
        "abc",  # Text
        "",  # Empty
        "2.0",  # Decimal representation of prime
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_small_prime_numbers_validation():
    """Test small prime numbers validation."""
    task = small_prime_numbers()

    # Valid small prime numbers (under 100)
    valid_cases = [
        "2",
        "3",
        "5",
        "7",
        "11",
        "13",
        "17",
        "19",
        "23",
        "29",
        "31",
        "37",
        "41",
        "43",
        "47",
        "53",
        "59",
        "61",
        "67",
        "71",
        "73",
        "79",
        "83",
        "89",
        "97",
    ]

    # Invalid cases
    invalid_cases = [
        "1",  # Not prime
        "4",
        "6",
        "8",
        "9",
        "10",  # Small composites
        "101",  # Prime but >= 100
        "103",  # Prime but >= 100
        "997",  # Large prime
        "100",  # Not prime and boundary
        " 7 ",  # With spaces
        "07",  # Leading zero
        "",  # Empty
        "abc",  # Text
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Prime Numbers Tasks Test ===")

    # Test general prime numbers
    print("\n--- Testing prime_numbers ---")
    task1 = prime_numbers()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_prime_numbers_validation()

    # Test small prime numbers
    print("\n--- Testing small_prime_numbers ---")
    task2 = small_prime_numbers()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_small_prime_numbers_validation()
