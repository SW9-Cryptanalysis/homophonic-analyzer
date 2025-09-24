import json
import os

def load_letter_frequencies(language: str = "english") -> dict[str, float]:
    """Loads letter frequency data for a specified language.

    Args:
        language (str): The language for which to load letter frequencies. Defaults to "english".

    Returns:
        dict[str, float]: A dictionary mapping letters to their frequencies.
    """
    language = language.lower()
    supported_languages = ["english"]
    
    if language not in supported_languages:
        raise ValueError("Unsupported language.")

    freq_path = os.path.join('data', 'frequencies', f'{language}_letter_frequencies.json')
    with open(freq_path, 'r') as f:
        frequency_data = json.load(f)
    
    return frequency_data