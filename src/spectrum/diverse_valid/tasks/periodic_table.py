"""
Periodic table elements generation tasks.

uv run python src/spectrum/diverse_valid/tasks/periodic_table.py
"""

import os

import pandas as pd

from spectrum.diverse_valid.generation_task import FunctionTask

elements_df = None


def load_elements_df():
    global elements_df
    if elements_df is None:
        elements_csv_path = "data/diverse_valid_data/periodic_table_elements.csv"

        if not os.path.exists(elements_csv_path):
            # Try to download with requests, ignoring SSL verification
            try:
                import requests

                url = "https://raw.githubusercontent.com/andrejewski/periodic-table/master/data.csv"
                r = requests.get(url, verify=False)
                os.makedirs(os.path.dirname(elements_csv_path), exist_ok=True)
                with open(elements_csv_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download periodic table data due to: {e}\n"
                    f"Please manually download https://github.com/andrejewski/periodic-table/blob/master/data.csv "
                    f"and place it as '{elements_csv_path}' in the current directory."
                )

        elements_df = pd.read_csv(elements_csv_path)
        # Clean up whitespace in column names and data
        elements_df.columns = elements_df.columns.str.strip()
        elements_df["symbol"] = elements_df["symbol"].str.strip()
        elements_df["name"] = elements_df["name"].str.strip()

    return elements_df


def validate_element_symbol(text: str) -> bool:
    """Validate that text is a chemical element symbol."""
    global elements_df
    if elements_df is None:
        load_elements_df()

    # Don't allow leading/trailing spaces
    if text != text.strip():
        return False

    # Get all valid symbols from the dataframe
    valid_symbols = elements_df["symbol"].tolist()

    return text in valid_symbols


def validate_element_name(text: str) -> bool:
    """Validate that text is a chemical element name."""
    global elements_df
    if elements_df is None:
        load_elements_df()

    # Don't allow leading/trailing spaces
    if text != text.strip():
        return False

    # Get all valid names from the dataframe (case insensitive)
    valid_names = elements_df["name"].str.lower().tolist()

    return text.lower() in valid_names


def element_symbols():
    """Chemical element symbols (H, He, Li, etc.)."""
    return FunctionTask(
        name="element_symbols",
        description="Chemical element symbol",
        examples=[
            "Sb",
            "He",
            "W",
        ],
        validation_fn=validate_element_symbol,
        max_new_tokens=5,
    )


def element_names():
    """Chemical element names (Hydrogen, Helium, Lithium, etc.)."""
    return FunctionTask(
        name="element_names",
        description="Chemical element name",
        examples=[
            "Niobium",
            "Antimony",
            "Iodine",
        ],
        validation_fn=validate_element_name,
        max_new_tokens=15,
    )


def test_element_symbols_validation():
    """Test element symbols validation."""
    task = element_symbols()

    # Valid element symbols
    valid_cases = [
        "H",  # Hydrogen
        "He",  # Helium
        "Li",  # Lithium
        "C",  # Carbon
        "N",  # Nitrogen
        "O",  # Oxygen
        "F",  # Fluorine
        "Ne",  # Neon
        "Na",  # Sodium
        "Mg",  # Magnesium
        "Al",  # Aluminum
        "Si",  # Silicon
        "P",  # Phosphorus
        "S",  # Sulfur
        "Cl",  # Chlorine
        "Fe",  # Iron
        "Cu",  # Copper
        "Zn",  # Zinc
        "Au",  # Gold
        "Pb",  # Lead
        "U",  # Uranium
    ]

    # Invalid cases
    invalid_cases = [
        "hydrogen",  # Full name, not symbol
        "h",  # Wrong case
        "HE",  # Wrong case
        "XX",  # Invalid symbol
        "Xy",  # Invalid symbol
        "",  # Empty
        "123",  # Numbers
        "H2",  # With number
        "H2O",  # Compound
        "NaCl",  # Compound
        " H ",  # With spaces
        "Carbon",  # Full name
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


def test_element_names_validation():
    """Test element names validation."""
    task = element_names()

    # Valid element names
    valid_cases = [
        "Hydrogen",
        "Helium",
        "Lithium",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Fluorine",
        "Neon",
        "Sodium",
        "Magnesium",
        "Aluminum",
        "Silicon",
        "Phosphorus",
        "Sulfur",
        "Chlorine",
        "Iron",
        "Copper",
        "Zinc",
        "Gold",
        "Lead",
        "Uranium",
        # Test case variations
        "hydrogen",  # Should work (case insensitive)
        "CARBON",  # Should work (case insensitive)
    ]

    # Invalid cases
    invalid_cases = [
        "H",  # Symbol, not name
        "He",  # Symbol, not name
        "Water",  # Compound, not element
        "Salt",  # Compound, not element
        "Steel",  # Alloy, not element
        "Air",  # Mixture, not element
        "",  # Empty
        "123",  # Numbers
        "Hydrogenium",  # Close but wrong
        "Carbonic",  # Close but wrong
        " Hydrogen ",  # With spaces (might be acceptable depending on implementation)
        "Element",  # Generic term
    ]

    # Run validation test
    success = task.test_validation(valid_cases, invalid_cases)
    return success


if __name__ == "__main__":
    print("=== Periodic Table Elements Tasks Test ===")

    # Test element symbols
    print("\n--- Testing element_symbols ---")
    task1 = element_symbols()
    print(f"Task: {task1.name}")
    print(f"Messages: {task1.messages}")
    test_element_symbols_validation()

    # Test element names
    print("\n--- Testing element_names ---")
    task2 = element_names()
    print(f"Task: {task2.name}")
    print(f"Messages: {task2.messages}")
    test_element_names_validation()
