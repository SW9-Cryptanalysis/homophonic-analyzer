from typing import List, Set
import numpy as np
from ..constants import FREQUENCY_TOLERANCE
from .pruning import prune_frequencies

def backtracking(
    cipher_frequencies: np.ndarray, 
    target_homophones: int, 
    target_sum: float, 
    max_candidates: int = 10
) -> List[Set[int]]:
    """
    Finds subsets of cipher symbols whose frequencies sum to a target value.
    Includes a pruning pre-processing step to improve performance.
    
    Args:
        cipher_frequencies: Array of dicts with 'symbol' and 'frequency' keys.
        target_homophones: Number of symbols to select.
        target_sum: Desired sum of the selected frequencies.
        max_candidates: Maximum number of candidate solutions to return.
    
    Returns:
        List of sets, each containing symbols that form a valid combination.
    """
    lower_bound = target_sum - FREQUENCY_TOLERANCE
    upper_bound = target_sum + FREQUENCY_TOLERANCE
    
    pruned_frequencies = prune_frequencies(
        cipher_frequencies, target_homophones, lower_bound, upper_bound
    )
    
    if len(pruned_frequencies) < target_homophones:
        return []
        
    candidate_results = []
    _backtrack(
        start_index=0,
        current_combination=set(),
        current_sum=0.0,
        pruned_frequencies=pruned_frequencies,
        target_homophones=target_homophones,
        target_sum=target_sum,
        bounds=(lower_bound, upper_bound),
        candidate_results=candidate_results
    )
    
    candidate_results.sort(key=lambda x: x[1])
    best_candidates = candidate_results[:max_candidates]
    return [candidate[0] for candidate in best_candidates]
    
    
def _backtrack(
    start_index: int,
    current_combination: set,
    current_sum: float,
    pruned_frequencies: np.ndarray,
    target_homophones: int,
    target_sum: float,
    bounds: tuple,
    candidate_results: list
):
    """
    The recursive helper function to find combinations.
    
    Args:
        start_index: Current index in pruned_frequencies to consider.
        current_combination: Set of currently selected symbols.
        current_sum: Sum of frequencies in the current combination.
        pruned_frequencies: The pruned array of frequency dicts.
        target_homophones: Number of symbols to select.
        target_sum: Desired sum of the selected frequencies.
        bounds: Tuple (lower_bound, upper_bound) for valid sums.
        candidate_results: List to store valid combinations and their distances.
    
    Returns:
        None. Results are appended to candidate_results.            
    """
    lower_bound, upper_bound = bounds
    n = len(pruned_frequencies)

    # Base case: Found a combination of the correct size
    if len(current_combination) == target_homophones:
        if lower_bound <= current_sum <= upper_bound:
            distance = abs(current_sum - target_sum)
            candidate_results.append((current_combination.copy(), distance))
        return

    # Pruning cases
    if start_index >= n or len(current_combination) > target_homophones:
        return

    for i in range(start_index, n):
        symbol = pruned_frequencies[i]['symbol']
        frequency = pruned_frequencies[i]['frequency']
        
        # Optimization: check before the recursive call
        if current_sum + frequency > upper_bound:
            break

        current_combination.add(symbol)
        _backtrack(
            i + 1, current_combination, current_sum + frequency,
            pruned_frequencies, target_homophones, target_sum,
            bounds, candidate_results
        )
        current_combination.remove(symbol)