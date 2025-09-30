from typing import Tuple
import math

from src.constants import RANGE_TOLERANCE, FEASIBILITY_THRESHOLD

def calculate_target_range(
    total_length: int, target_freq: float
) -> Tuple[int, int]:
    """Calculates the min and max expected count for a letter.
    
    Args: 
        total_length: Total number of symbols in the cipher.
        target_freq: Target frequency for the letter (between 0 and 1).
        
    Returns:
        A tuple (min_count, max_count) representing the expected range.
    """
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
    
    Args:
        candidate_pool_size: Number of available cipher symbols.
        homophones_needed: Number of homophones required for the letter.
    Returns:
        True if the number of combinations is below the threshold, False otherwise.
    """
    # math.comb(n, k) efficiently calculates "n choose k"
    num_combinations = math.comb(candidate_pool_size, homophones_needed)
    return num_combinations <= FEASIBILITY_THRESHOLD