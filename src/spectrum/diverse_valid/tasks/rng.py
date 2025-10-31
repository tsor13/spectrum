"""
uv run src/spectrum/diverse_valid/tasks/rng.py
"""

from spectrum.diverse_valid.generation_task import FunctionTask, InclusionTask


def rng_1_10():
    return InclusionTask(
        name="rng_1_10",
        description="Generate a number between 1 and 10.",
        examples=["3", "7", "10"],
        valid_strings=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        max_new_tokens=4,
    )


def rng_1_100():
    def validate_number_1_100(text: str) -> bool:
        """Validate that text is a number between 1 and 100."""
        # Don't allow leading/trailing spaces or leading zeros
        if text != text.strip():
            return False
        if text.startswith("0") and len(text) > 1:
            return False

        try:
            num = int(text)
            return num >= 1 and num <= 100
        except ValueError:
            return False

    return FunctionTask(
        name="rng_1_100",
        description="Generate a number between 1 and 100.",
        examples=[
            "35",
            "94",
            "71",
        ],
        validation_fn=validate_number_1_100,
        max_new_tokens=4,
    )


# # Keep old name for backwards compatibility
# def rng_10():
#     return rng_1_10()
#
# def test_rng_1_10_validation():
#     """Test rng_1_10 task validation."""
#     task = rng_1_10()
#
#     # Valid cases (1-10)
#     valid_cases = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
#
#     # Invalid cases
#     invalid_cases = [
#         "0",           # Below range
#         "11",          # Above range
#         "-1",          # Negative
#         "one",         # Text number
#         "1.5",         # Decimal
#         " 5 ",         # With spaces
#         "",            # Empty
#         "5\n",         # With newline
#         "05",          # Leading zero
#     ]
#
#     # Run validation test
#     success = task.test_validation(valid_cases, invalid_cases)
#     return success
#
# def test_rng_1_100_validation():
#     """Test rng_1_100 task validation."""
#     task = rng_1_100()
#
#     # Valid cases (1-100)
#     valid_cases = ["1", "10", "50", "99", "100"]
#
#     # Invalid cases
#     invalid_cases = [
#         "0",           # Below range
#         "101",         # Above range
#         "-1",          # Negative
#         "one",         # Text number
#         "50.5",        # Decimal
#         " 50 ",        # With spaces (should be invalid for this task)
#         "",            # Empty
#         "050",         # Leading zero
#         "999",         # Way above range
#     ]
#
#     # Run validation test
#     success = task.test_validation(valid_cases, invalid_cases)
#     return success


if __name__ == "__main__":
    print("=== RNG Tasks Test ===")

    # Test 1-10 range
    print("\n--- Testing rng_1_10 ---")
    task1 = rng_1_10()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_rng_1_10_validation()

    # Test 1-100 range
    print("\n--- Testing rng_1_100 ---")
    task2 = rng_1_100()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_rng_1_100_validation()

    # Test backwards compatibility
    print("\n--- Testing rng_10 (backwards compatibility) ---")
    task3 = rng_10()
    print(f"Task: {task3.name} (should be rng_1_10)")
