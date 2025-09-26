from typing import List, Set, Tuple
import numpy as np

from ..load_cipher import load_cipher, get_cipher_frequencies
from ..constants import EXAMPLE_CIPHERS_PATH
from ..utils import load_letter_frequencies
from .feasibility import calculate_target_range, is_feasible
from .backtracking import backtracking

def find_letter_candidates():
    cipher = load_cipher(EXAMPLE_CIPHERS_PATH / "cipher.json")
    cipher_frequencies = get_cipher_frequencies(cipher)
    letter_frequencies = load_letter_frequencies("english")
    print(f"Loaded cipher with {(cipher_frequencies)} unique symbols")

    # Go through each letter in letter frequencies starting from the least frequent
    sorted_letters = sorted(letter_frequencies.items(), key=lambda item: item[1])
    
    for letter, freq in sorted_letters:
        print(f"Analyzing letter: {letter} with frequency {freq}")
        
        min_max_range = calculate_target_range(len(cipher_frequencies), freq)
        print(f"  Expected range in cipher: {min_max_range}")
        
        # Checks if the max number of homophones is feasible
        is_letter_feasible = is_feasible(len(cipher_frequencies), 9)
        if not is_letter_feasible:
            continue
        
        # Find all subsets of cipher symbols whose frequencies sum to the target range
        candidates = []
        for i in range(min_max_range[0], min_max_range[1] + 1):
            print(f"  Finding candidates with {i} homophones and frequency {freq}")
            candidates += backtracking(
                cipher_frequencies,
                i,
                freq
            )
        print(f"  Found {len(candidates)} candidate sets for letter '{letter}'")
        