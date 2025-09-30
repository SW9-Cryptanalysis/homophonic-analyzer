import json
import numpy as np
import pathlib
from collections import Counter
import os
from .constants import EXAMPLE_CIPHERS_PATH

def load_cipher(filepath: pathlib.Path) -> list[int]:
    """Reads a cipher from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        The ciphertext as a list of numbers.
    """
    cipher_path = os.path.join(EXAMPLE_CIPHERS_PATH, filepath)
    with open(cipher_path, 'r') as f:
        data = json.load(f)
    ciphertext_str = data['ciphertext']
    return [int(num) for num in ciphertext_str.split()]


def get_cipher_frequencies(cipher_text: list[int]) -> np.ndarray:
    """Calculates the frequency of each symbol in the cipher text.

    Args:
        cipher_text: A list of numbers representing the ciphertext.

    Returns:
        A numpy structured array with 'symbol' and 'frequency' fields.
    """
    counts = Counter(cipher_text)
    print(f"  Unique symbols in cipher: {len(counts)}")
    total = len(cipher_text)
    print(f"  Total symbols in cipher: {total}")
    
    for symbol, count in counts.items():
        if symbol == 62:
            print(f"   Symbol 62 count: {count}")
    
    freq_data = [(symbol, count / total) for symbol, count in counts.items()]
    print(f"  Sample frequencies: {freq_data[:5]}")
    structured_array = np.array(freq_data, dtype=[('symbol', int), ('frequency', float)])
    print(f"  Structured array sample: {structured_array[14]}")
    
    return structured_array
