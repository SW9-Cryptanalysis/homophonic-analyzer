import math
from typing import List, Set

from ..load_cipher import load_cipher, get_cipher_frequencies
from ..constants import EXAMPLE_CIPHERS_PATH
from ..utils import load_letter_frequencies
from .feasibility import calculate_target_range, is_feasible
from .backtracking import backtracking

def find_letter_candidates(cipher_file_path: str = "cipher-1.json") -> List[Set[int]]:
    cipher = load_cipher(EXAMPLE_CIPHERS_PATH / cipher_file_path)
    cipher_frequencies = get_cipher_frequencies(cipher)
    letter_frequencies = load_letter_frequencies("english")
    print(f"Length of cipher: {len(cipher)}")
    print(f"Loaded cipher with {len(cipher_frequencies)} unique symbols")

    # Go through each letter in letter frequencies starting from the least frequent
    sorted_letters = sorted(letter_frequencies.items(), key=lambda item: item[1])
    
    for letter, freq in sorted_letters:
        print(f"\n *** Analyzing letter: {letter} with frequency {freq} ***")
        
        min_max_range = calculate_target_range(len(cipher_frequencies), freq)
        print(f"  Expected range in cipher: {min_max_range}")
        
        # Checks if the max number of homophones is feasible
        is_letter_feasible = is_feasible(len(cipher_frequencies), min_max_range[1])
        if not is_letter_feasible:
            continue
        
        # Find all subsets of cipher symbols whose frequencies sum to the target range
        candidates = []
        for i in range(min_max_range[0], min_max_range[1] + 1):
            print(f"\n  Finding candidates with {i} homophones")
            candidates += backtracking(
                cipher_frequencies,
                i,
                freq
            )
        
        print_candidates_clean(candidates)
        
        num_combinations = math.comb(len(cipher_frequencies), min_max_range[1])
        print(f"  Percentage of combinations removed: {(num_combinations - len(candidates)) / num_combinations * 100:.6f}%")
    
    return candidates


def print_candidates_clean(candidates: List[Set[int]]):
    clean_candidates = []
    for candidate_set in candidates:
        clean_set = {int(symbol) for symbol in candidate_set}
        clean_candidates.append(clean_set)
    
    if clean_candidates:
        print("  Sample candidates:")
        for i, candidate in enumerate(clean_candidates):
            sorted_candidate = sorted(candidate)
            print(f"    {i+1}: {{{', '.join(map(str, sorted_candidate))}}}")