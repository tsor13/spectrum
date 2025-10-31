"""
Email address generation tasks.

uv run python src/spectrum/diverse_valid/tasks/emails.py
"""

import re

from spectrum.diverse_valid.generation_task import FunctionTask


def validate_basic_email(text: str) -> bool:
    """Validate basic email format: user@domain.tld"""
    text = text.strip()

    # Basic email pattern: one or more chars, @, domain, ., tld
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if not re.match(pattern, text):
        return False

    # Additional checks
    parts = text.split("@")
    if len(parts) != 2:
        return False

    local_part, domain_part = parts

    # Local part checks
    if len(local_part) == 0 or len(local_part) > 64:
        return False
    if local_part.startswith(".") or local_part.endswith("."):
        return False
    if ".." in local_part:
        return False

    # Domain part checks
    if len(domain_part) == 0 or len(domain_part) > 255:
        return False
    if domain_part.startswith(".") or domain_part.endswith("."):
        return False
    if ".." in domain_part:
        return False

    return True


def validate_professional_email(text: str) -> bool:
    """Validate professional email (common business domains)."""
    if not validate_basic_email(text):
        return False

    text = text.lower().strip()

    # List of common professional domains
    professional_domains = {
        "gmail.com",
        "outlook.com",
        "hotmail.com",
        "yahoo.com",
        "company.com",
        "corp.com",
        "business.com",
        "office.com",
        "work.com",
        "enterprise.com",
        "organization.org",
        "institution.edu",
        "government.gov",
        "research.edu",
        "university.edu",
        "college.edu",
    }

    domain = text.split("@")[1]

    # Allow any .com, .org, .edu, .gov domains as professional
    if (
        domain.endswith(".com")
        or domain.endswith(".org")
        or domain.endswith(".edu")
        or domain.endswith(".gov")
    ):
        return True

    return domain in professional_domains


def basic_emails():
    """Basic email addresses with standard format."""
    return FunctionTask(
        name="basic_emails",
        description="Email address",
        examples=[
            "tsor13@cs.washington.edu",
            "alex.jones@domain.net",
            "itsagoodday@gmail.com",
        ],
        validation_fn=validate_basic_email,
        max_new_tokens=40,
    )


def professional_emails():
    """Professional email addresses (business domains)."""
    return FunctionTask(
        name="professional_emails",
        description="Generate a professional email address.",
        examples=[
            "tsor13@cs.washington.edu",
            "sarah.johannesburg@organization.org",
            "yash@anthropic.com",
        ],
        validation_fn=validate_professional_email,
        max_new_tokens=40,
    )


def test_basic_email_validation():
    """Test basic email validation."""
    task = basic_emails()

    # Valid email formats
    valid_cases = [
        "user@example.com",  # Basic format
        "john.doe@company.org",  # With dot in name
        "test123@domain.net",  # With numbers
        "tsor13@cs.washington.edu",  # With numbers
        "user+tag@example.com",  # With plus
        "user_name@example.com",  # With underscore
        "user-name@example.com",  # With hyphen
        "a@b.co",  # Minimal valid
        "very.long.email.address@very.long.domain.example.com",  # Long but valid
    ]

    # Invalid cases
    invalid_cases = [
        "userexample.com",  # Missing @
        "@example.com",  # Missing local part
        "user@",  # Missing domain
        "user@.com",  # Domain starts with dot
        "user@example.",  # Domain ends with dot
        "user@example",  # Missing TLD
        ".user@example.com",  # Local starts with dot
        "user.@example.com",  # Local ends with dot
        "us..er@example.com",  # Double dot in local
        "user@ex..ample.com",  # Double dot in domain
        "user name@example.com",  # Space in local part
        "user@exam ple.com",  # Space in domain
        "",  # Empty
        "user@",  # Incomplete
        "user@example.c",  # TLD too short
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_professional_email_validation():
    """Test professional email validation."""
    task = professional_emails()

    # Valid professional email formats
    valid_cases = [
        "john.smith@company.com",  # Standard business
        "sarah@organization.org",  # Organization
        "mike.jones@business.com",  # Business domain
        "admin@university.edu",  # Educational
        "contact@government.gov",  # Government
        "user@mycorp.com",  # Any .com domain
        "info@nonprofit.org",  # Any .org domain
    ]

    # Invalid cases (either bad format or non-professional domains)
    invalid_cases = [
        "userexample.com",  # Bad format
        "user@example",  # No TLD
        "user@personal.xyz",  # Non-professional TLD
        "user@test.info",  # Non-professional TLD
        "",  # Empty
        "user@domain.co.uk",  # Non-standard TLD (though could be professional)
        "@company.com",  # Missing local part
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Email Tasks Test ===")

    # Test basic emails
    print("\n--- Testing basic_emails ---")
    task1 = basic_emails()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_basic_email_validation()

    # Test professional emails
    print("\n--- Testing professional_emails ---")
    task2 = professional_emails()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_professional_email_validation()
