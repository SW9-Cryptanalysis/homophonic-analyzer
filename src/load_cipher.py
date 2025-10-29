import json
import numpy as np
import pathlib
from collections import Counter
import os
from .utils.constants import EXAMPLE_CIPHERS_PATH

def load_cipher(filepath: pathlib.Path) -> list[int]:
    """Read a cipher from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        The ciphertext as a list of numbers.

    """
    cipher_path = os.path.join(EXAMPLE_CIPHERS_PATH, filepath)
    with open(cipher_path) as f:
        data = json.load(f)
    ciphertext_str = data["ciphertext"]
    return [int(num) for num in ciphertext_str.split()]


def get_cipher_frequencies(cipher_text: list[int]) -> np.ndarray:
    """Calculate the frequency of each symbol in the cipher text.

    Args:
        cipher_text: A list of numbers representing the ciphertext.

    Returns:
        A numpy structured array with 'symbol' and 'frequency' fields.

    """
    counts = Counter(cipher_text)
    total = len(cipher_text)

    freq_data = [(symbol, count / total) for symbol, count in counts.items()]
    structured_array = np.array(freq_data, dtype=[("symbol", int),
                                                  ("frequency", float)])

    return structured_array
