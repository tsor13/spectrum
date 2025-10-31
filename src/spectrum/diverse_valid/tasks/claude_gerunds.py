"""
uv run src/spectrum/diverse_valid/tasks/claude_gerunds.py
"""

from io import StringIO

from spectrum.diverse_valid.generation_task import FunctionTask

gerund_df = None


def load_gerund_df():
    global gerund_df
    if gerund_df is None:
        import os

        import pandas as pd
        import requests

        gerund_path = "data/diverse_valid_data/en-verbs.txt"
        if not os.path.exists(gerund_path):
            try:
                url = "https://raw.githubusercontent.com/gutfeeling/word_forms/master/word_forms/en-verbs.txt"
                r = requests.get(url, verify=False)
                os.makedirs(os.path.dirname(gerund_path), exist_ok=True)
                with open(gerund_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download verbs list due to: {e}\n"
                    f"Please manually download https://github.com/gutfeeling/word_forms/blob/master/word_forms/en-verbs.txt "
                    f"and place it as '{gerund_path}' in the current directory."
                )

        # Read verbs and convert to gerunds
        with open(gerund_path) as f:
            lines = f.readlines()[19:]
            lines_str = "".join(lines)
            # gerund_df = pd.read_csv(StringIO(lines_str))
            # no header
            gerund_df = pd.read_csv(StringIO(lines_str), header=None)

    return gerund_df


# validation function is whether or not it's a number between 0 and 2**32 - 1
def is_valid_gerund(x: str) -> bool:
    # check if ends with ing, alphabet, and one word
    # return x.endswith("ing") and x.isalpha() and len(x.split()) == 1
    # check if in gerund_df
    if gerund_df is None:
        load_gerund_df()
    gerunds = gerund_df[5].tolist()
    return x.strip().lower() in gerunds


def claude_gerunds():
    return FunctionTask(
        name="claude_gerunds",
        description="Generate an English gerund ending in -ing.",
        examples=["Schlepping", "Hoisting", "Thinking"],
        validation_fn=is_valid_gerund,
        max_new_tokens=4,
    )


def test_claude_gerunds_validation():
    """Test Claude gerunds validation with comprehensive cases."""
    task = claude_gerunds()

    # Valid gerund cases (from examples + common ones)
    valid_cases = [
        "thinking",  # Common gerund
        "running",  # Common gerund
        "walking",  # Common gerund
        "eating",  # Common gerund
        "sleeping",  # Common gerund
    ]

    # Invalid cases
    invalid_cases = [
        "think",  # Base verb, not gerund
        "thoughts",  # Noun, not gerund
        "quickly",  # Adverb ending in -ly
        "running fast",  # Multiple words
        "123",  # Number
        "",  # Empty
        "runing",  # Misspelled
        "to run",  # Infinitive
        "ran",  # Past tense
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Claude Gerunds Task Test ===")
    task = claude_gerunds()
    print(f"Task: {task.name}")
    print(f"Messages: {task.messages}")
    print()

    # Run validation tests
    print("Running validation tests...")
    test_claude_gerunds_validation()
