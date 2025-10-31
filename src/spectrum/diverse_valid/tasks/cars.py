"""
uv run src/spectrum/diverse_valid/tasks/cars.py
"""

from spectrum.diverse_valid.generation_task import FunctionTask


# validation function is whether or not it's a number between 0 and 2**32 - 1
def is_valid_seed(x: str) -> bool:
    try:
        seed = int(x)
        return 0 <= seed < 2**32
    except ValueError:
        return False


car_df = None


def load_car_df():
    global car_df
    import os

    import pandas as pd

    car_df = pd.DataFrame()
    dfs = []
    for file in os.listdir("data/diverse_valid_data/us-car-models-data/"):
        if file.endswith(".csv"):
            df = pd.read_csv(f"data/diverse_valid_data/us-car-models-data/{file}")
            dfs.append(df)
    car_df = pd.concat(dfs)
    return car_df


def validate_car_make_model(text: str) -> bool:
    global car_df
    if car_df is None:
        load_car_df()
    # check if both make and model in the string
    for row in car_df.itertuples():
        if row.make.lower() in text.lower() and row.model.lower() in text.lower():
            return True
    return False


def car_make_model():
    return FunctionTask(
        name="car_make_model",
        description="Car make and model.",
        examples=[
            "Acura Integra",
            "Ford Mustang",
            "Tesla Model 3",
        ],
        validation_fn=validate_car_make_model,
        max_new_tokens=16,
    )


def validate_car_brand(text: str) -> bool:
    global car_df
    if car_df is None:
        load_car_df()
    # check if make in the string
    makes = car_df["make"].unique().tolist()
    makes = [make.lower() for make in makes]
    return text.strip().lower() in makes


def car_brand():
    return FunctionTask(
        name="car_brand",
        description="Car brand.",
        examples=[
            "Acura",
            "Ford",
            "Tesla",
        ],
        validation_fn=validate_car_brand,
        max_new_tokens=8,
    )


def test_car_make_model_validation():
    """Test car make+model validation."""
    task = car_make_model()

    # Valid make+model combinations
    valid_cases = ["Tesla Model 3", "Toyota Corolla", "Audi A8 1997"]

    # Invalid cases for make+model
    invalid_cases = [
        "Tesla",  # Brand only, needs model
        "tht",  # Random text
        "invalid",  # Not real
        "thgreen",  # Random text
        "2001",  # Year only
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_car_brand_validation():
    """Test car brand validation."""
    task = car_brand()

    # Valid car brands
    valid_cases = ["Tesla", "Toyota", "Audi"]

    # Invalid cases for brands
    invalid_cases = [
        "Tesla Model 3",  # Too specific, includes model
        "Toyota Corolla",  # Too specific, includes model
        "Audi A8 1997",  # Too specific, includes model
        "Boeing",  # Not a car brand
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Car Tasks Test ===")

    # Test car make+model
    print("\n--- Testing car_make_model ---")
    task1 = car_make_model()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_car_make_model_validation()

    # Test car brand
    print("\n--- Testing car_brand ---")
    task2 = car_brand()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_car_brand_validation()
