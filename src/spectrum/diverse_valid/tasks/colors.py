"""
uv run src/spectrum/diverse_valid/tasks/colors.py
"""

from spectrum.diverse_valid.generation_task import FunctionTask


# validation function is whether or not it's a number between 0 and 2**32 - 1
def is_valid_seed(x: str) -> bool:
    try:
        seed = int(x)
        return 0 <= seed < 2**32
    except ValueError:
        return False


color_df = None


def load_color_df():
    global color_df
    if color_df is None:
        # color_df = pd.read_csv("src/spectrum/diverse_valid/data/colornames.csv")
        # get a comprehensive list of colors from the web
        # If SSL certificate verification fails, try to download the file manually first.
        import os

        import pandas as pd

        color_csv_path = "data/diverse_valid_data/colornames.csv"
        if not os.path.exists(color_csv_path):
            # Try to download with requests, ignoring SSL verification
            try:
                import requests

                url = "https://raw.githubusercontent.com/meodai/color-names/main/src/colornames.csv"
                r = requests.get(url, verify=False)
                with open(color_csv_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download color names CSV due to: {e}\n"
                    f"Please manually download https://github.com/meodai/color-names/blob/main/src/colornames.csv "
                    f"and place it as '{color_csv_path}' in the current directory."
                )
        color_df = pd.read_csv(color_csv_path)
    # filter to only goodname
    # color_df = color_df[color_df["good name"] == 'x']
    return color_df


def validate_color(text: str) -> bool:
    global color_df
    if color_df is None:
        load_color_df()
    colors = color_df["name"].tolist()
    # to lower
    colors = [color.lower() for color in colors]
    # also add individual words
    # single_words = [word for word in text.strip().lower().split() for text in colors]
    # colors = colors + single_words
    return text.strip().lower() in colors


# def color_noex():
#     return FunctionTask(
#         name="color_noex",
#         description="Generate a color name.",
#         examples=[
#             "Red",
#             "Green",
#             "Blue",
#         ],
#         validation_fn=validate_color,
#         max_new_tokens=8,
#     )


def color_normal_ex():
    return FunctionTask(
        name="color_normal_ex",
        description="Generate a color name.",
        examples=["Green", "Red", "White"],
        validation_fn=validate_color,
        max_new_tokens=8,
    )


def color_interesting_ex():
    return FunctionTask(
        name="color_interesting_ex",
        description="Generate a color name.",
        examples=["Otterly Brown", "Petal Pink", "Cherry"],
        validation_fn=validate_color,
        max_new_tokens=8,
    )


def test_color_validation():
    """Test color validation with comprehensive cases."""
    # Test multiple color tasks
    task_factories = [
        # ("color_noex", color_noex),
        ("color_normal_ex", color_normal_ex),
        ("color_interesting_ex", color_interesting_ex),
    ]

    for task_name, task_factory in task_factories:
        task = task_factory()  # Create task only when function is called
        print(f"\n=== Testing {task_name} ===")

        # Valid color cases
        valid_cases = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "pink",
            "brown",
        ]

        # Invalid cases
        invalid_cases = [
            "notacolor",  # Not a color
            "123",  # Number
            "",  # Empty
            " ",  # Space only
            "blueish",  # Close but not exact
            "car",  # Random word
        ]

        # Run validation test
        success = task.test_validation(valid_cases, invalid_cases)
        if success:
            print(f"✅ {task_name} validation passed!")
        else:
            print(f"❌ {task_name} validation had issues")


if __name__ == "__main__":
    # Show task info
    task = color_normal_ex()
    print(f"Example task: {task.name}")
    print(f"Messages: {task.messages}")

    # Run validation tests
    test_color_validation()
