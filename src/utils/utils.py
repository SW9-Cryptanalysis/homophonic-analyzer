import json
import os
from .constants import FREQUENCIES_PATH

def load_letter_frequencies(language: str = "english") -> dict[str, float]:
    """Load letter frequency data for a specified language.

    Args:
        language (str): The language for which to load letter frequencies. Defaults to
				"english".

    Returns:
        dict[str, float]: A dictionary mapping letters to their frequencies.

    """
    language = language.lower()
    supported_languages = ["english"]

    if language not in supported_languages:
        raise ValueError("Unsupported language.")

    freq_path = os.path.join(FREQUENCIES_PATH, f"{language}_letter_frequencies.json")
    with open(freq_path) as f:
        frequency_data = json.load(f)

    return frequency_data
