"""
Phone number generation tasks.

uv run python src/spectrum/diverse_valid/tasks/phone_numbers.py
"""

import re

from spectrum.diverse_valid.generation_task import FunctionTask


def validate_us_phone_number(text: str) -> bool:
    """Validate US phone number in format XXX-XXX-XXXX or (XXX) XXX-XXXX."""
    text = text.strip()

    # Pattern 1: XXX-XXX-XXXX
    pattern1 = r"^\d{3}-\d{3}-\d{4}$"

    # Pattern 2: (XXX) XXX-XXXX
    pattern2 = r"^\(\d{3}\) \d{3}-\d{4}$"

    # pattern 3: XXX XXX XXXX
    pattern3 = r"^\d{3} \d{3} \d{4}$"

    return bool(
        re.match(pattern1, text) or re.match(pattern2, text) or re.match(pattern3, text)
    )


def validate_international_phone(text: str) -> bool:
    """Validate international phone number with country code (+X or +XX followed by digits)."""
    text = text.strip()

    # Must start with +
    if not text.startswith("+"):
        return False

    # Remove + and count digits vs separators
    rest = text[1:]
    digits_only = re.sub(r"[^\d]", "", rest)

    # Must have 7-15 total digits
    if not (7 <= len(digits_only) <= 15):
        return False

    # Pattern: +countrycode followed by digits, spaces, hyphens allowed
    # Allow reasonable formatting but not too many separators
    pattern = r"^\+\d{1,3}([-\s]?\d+)+$"

    if not re.match(pattern, text):
        return False

    # Check that separators aren't excessive (no more than 6 separators total)
    separators = text.count(" ") + text.count("-")
    if separators > 6:
        return False

    return True


def us_phone_numbers():
    """US phone numbers in standard formats."""
    return FunctionTask(
        name="us_phone_numbers",
        description="US phone number",
        examples=[
            "(210) 346-0967",
            "646 562-1938",
            "800-895-0522",
        ],
        validation_fn=validate_us_phone_number,
        max_new_tokens=20,
    )


def international_phone_numbers():
    """International phone numbers with country codes."""
    return FunctionTask(
        name="international_phone_numbers",
        description="International phone number with country code.",
        examples=[
            "+1 413-121-2591",
            "+44 10 2958 3938",
            "+81 3 8328 5625",
        ],
        validation_fn=validate_international_phone,
        max_new_tokens=25,
    )


def test_us_phone_validation():
    """Test US phone number validation."""
    task = us_phone_numbers()

    # Valid US phone formats
    valid_cases = [
        "555-123-4567",  # Standard format
        "800-555-1234",  # Toll free
        "(555) 123-4567",  # Parentheses format
        "(800) 555-1234",  # Toll free with parentheses
        "212-555-0123",  # NYC area code
        "212 555 0123",  # NYC area code
    ]

    # Invalid cases
    invalid_cases = [
        "555-1234",  # Too short
        "555-123-45678",  # Too long
        "55-123-4567",  # Area code too short
        "555-12-4567",  # Exchange too short
        "+1 555-123-4567",  # International format
        "555.123.4567",  # Dots instead of dashes
        "(555)123-4567",  # No space after parentheses
        "abc-def-ghij",  # Letters
        "",  # Empty
        "1234567890",  # No formatting
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_international_phone_validation():
    """Test international phone number validation."""
    task = international_phone_numbers()

    # Valid international formats
    valid_cases = [
        "+1 555-123-4567",  # US with country code
        "+44 20 7946 0958",  # UK
        "+33 1 42 68 53 00",  # France
        "+49 30 12345678",  # Germany
        "+81 3 1234 5678",  # Japan
        "+86 10 1234 5678",  # China
        "+91 11 2345 6789",  # India
    ]

    # Invalid cases
    invalid_cases = [
        "555-123-4567",  # No country code
        "+1234567890123456",  # Too many digits
        "+1 555",  # Too few digits
        "1 555-123-4567",  # Missing +
        "+",  # Just +
        "+abc 123-4567",  # Letters in country code
        "",  # Empty
        "+1",  # Just country code
        "+999 999 999 999 9999",  # Too many digits (16)
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Phone Number Tasks Test ===")

    # Test US phone numbers
    print("\n--- Testing us_phone_numbers ---")
    task1 = us_phone_numbers()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_us_phone_validation()

    # Test international phone numbers
    print("\n--- Testing international_phone_numbers ---")
    task2 = international_phone_numbers()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_international_phone_validation()
