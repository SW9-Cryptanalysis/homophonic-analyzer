from typing import List, Set
import numpy as np
from ..constants import FREQUENCY_TOLERANCE

def backtracking(cipher_frequencies: np.ndarray, target_homophones: int, target_sum: float, max_candidates: int = 10) -> List[Set[int]]:
    """Finds subsets of cipher symbols whose frequencies sum to a target value using backtracking.
    
    Args:
        cipher_frequencies (np.ndarray): Array of cipher symbols with their frequencies.
        target_homophones (int): Number of homophones to select.
        target_sum (float): Target frequency sum for the selected homophones.
        max_candidates (int): Maximum number of candidate sets to return.
        
    Returns:
        List[Set[int]]: List of sets of cipher symbols that are candidates for the letter.
    """
    candidate_results = []
    n = len(cipher_frequencies)
    
    lower_bound = target_sum - FREQUENCY_TOLERANCE
    upper_bound = target_sum + FREQUENCY_TOLERANCE
    
    def backtrack(start_index: int, current_combination: Set[int], current_sum: float):
        if len(current_combination) == target_homophones:
            if lower_bound <= current_sum <= upper_bound:
                distance = abs(current_sum - target_sum)
                candidate_results.append((current_combination.copy(), current_sum, distance))
            return
        
        if len(current_combination) > target_homophones:
            return
            
        if current_sum > upper_bound:
            return
        
        for i in range(start_index, n):
            symbol = cipher_frequencies[i]['symbol']
            frequency = cipher_frequencies[i]['frequency']
            
            current_combination.add(symbol)
            
            backtrack(i + 1, current_combination, current_sum + frequency)
            
            current_combination.remove(symbol)
    
    # Start backtracking from index 0 with empty combination
    backtrack(0, set(), 0.0)
    
    candidate_results.sort(key=lambda x: x[2])
    best_candidates = candidate_results[:max_candidates]
    results = [candidate[0] for candidate in best_candidates]
    
    return results
    