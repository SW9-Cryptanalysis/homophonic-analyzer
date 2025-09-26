from typing import List, Set
import numpy as np
from ..constants import FREQUENCY_TOLERANCE

# Given the cipher frequencies as a ndarray, target homophones as integer, and a target sum as float
def backtracking(cipher_frequencies: np.ndarray, target_homophones: int, target_sum: float) -> List[Set[int]]:
    results = []
    n = len(cipher_frequencies)
    print(f"  Starting backtracking with {n} items, target_homophones={target_homophones}, target_sum={target_sum}")
    
    # Calculate tolerance bounds
    lower_bound = target_sum - FREQUENCY_TOLERANCE
    upper_bound = target_sum + FREQUENCY_TOLERANCE
    
    def backtrack(start_index: int, current_combination: Set[int], current_sum: float):
        # Base case: we have selected exactly target_homophones symbols
        if len(current_combination) == target_homophones:
            # Check if the sum is within tolerance
            if lower_bound <= current_sum <= upper_bound:
                results.append(current_combination.copy())
            return
        
        # Pruning: if we already have too many symbols, stop
        if len(current_combination) > target_homophones:
            return
            
        # Pruning: if current sum already exceeds upper bound, stop this path
        if current_sum > upper_bound:
            return
        
        # Try each symbol starting from start_index
        for i in range(start_index, n):
            symbol = cipher_frequencies[i]['symbol']
            frequency = cipher_frequencies[i]['frequency']
            
            # Add current symbol to combination
            current_combination.add(symbol)
            
            # Recursive call with updated state
            backtrack(i + 1, current_combination, current_sum + frequency)
            
            # Backtrack: remove the symbol from combination
            current_combination.remove(symbol)
    
    # Start backtracking from index 0 with empty combination
    backtrack(0, set(), 0.0)
    
    print(f"  Found {len(results)} valid combinations")
    return results
    