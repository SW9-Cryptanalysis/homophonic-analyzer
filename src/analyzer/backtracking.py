from typing import List, Set
import numpy as np
from ..constants import FREQUENCY_TOLERANCE

# Given the cipher frequencies as a ndarray, target homophones as integer, and a target sum as float
def backtracking(cipher_frequencies: np.ndarray, target_homophones: int, target_sum: float, max_candidates: int = 10) -> List[Set[int]]:
    # Store results with their distances from target
    candidate_results = []
    n = len(cipher_frequencies)
    print(f"  Starting backtracking with {n} items, target_homophones={target_homophones}, target_sum={target_sum}")
    
    # Calculate tolerance bounds
    lower_bound = target_sum - FREQUENCY_TOLERANCE
    upper_bound = target_sum + FREQUENCY_TOLERANCE
    
    def backtrack(start_index: int, current_combination: Set[int], current_sum: float):
        # Base case: we have selected exactly target_homophones symbols
        if len(current_combination) == target_homophones:
            # Only consider combinations within tolerance bounds
            if lower_bound <= current_sum <= upper_bound:
                distance = abs(current_sum - target_sum)
                candidate_results.append((current_combination.copy(), current_sum, distance))
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
    
    # Sort candidates by distance from target (best first)
    candidate_results.sort(key=lambda x: x[2])
    
    # Keep only the top max_candidates
    best_candidates = candidate_results[:max_candidates]
    
    # Extract just the combinations from the best candidates
    results = [candidate[0] for candidate in best_candidates]
    
    print(f"  Found {len(candidate_results)} total candidates, keeping top {len(results)} closest to target")
    if results:
        best_distance = best_candidates[0][2]
        worst_distance = best_candidates[-1][2] if len(best_candidates) > 1 else best_distance
        print(f"  Best distance from target: {best_distance:.6f}, worst distance: {worst_distance:.6f}")
    
    return results
    