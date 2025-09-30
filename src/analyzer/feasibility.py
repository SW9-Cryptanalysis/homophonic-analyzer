from typing import Tuple
import math

from src.constants import RANGE_TOLERANCE, FEASIBILITY_THRESHOLD

def calculate_target_range(
    total_length: int, target_freq: float
) -> Tuple[int, int]:
    """Calculates the min and max expected count for a letter."""
    expected_count = total_length * target_freq
    delta = total_length * RANGE_TOLERANCE
    min_count = math.floor(expected_count - delta)
    max_count = math.ceil(expected_count + delta)
    return max(1, min_count), max_count

def is_feasible(candidate_pool_size: int, homophones_needed: int) -> bool:
    """
    Checks if the number of combinations is below the feasibility threshold.
    The number of combinations is nCr, where n is the number of candidate ciphers
    and r is the number of homophones for the letter.
    """
    # math.comb(n, k) efficiently calculates "n choose k"
    num_combinations = math.comb(candidate_pool_size, homophones_needed)
    print(f"  Maximum number of combinations: {num_combinations}")
    return num_combinations <= FEASIBILITY_THRESHOLD