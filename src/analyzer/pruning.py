import numpy as np

def prune_frequencies(
    cipher_frequencies: np.ndarray, 
    target_homophones: int, 
    lower_bound: float, 
    upper_bound: float
) -> np.ndarray:
    """
    Filters out frequencies that cannot possibly be part of a valid solution.
    
    Args:
        cipher_frequencies: Array of dicts with 'symbol' and 'frequency'.
        target_homophones: Number of homophones to select.
        lower_bound: Minimum acceptable sum of frequencies.
        upper_bound: Maximum acceptable sum of frequencies.
        
    Returns:
        np.ndarray: Pruned array of frequency dicts.
    """
    if len(cipher_frequencies) < target_homophones:
        return np.array([])

    # Sort frequencies once to easily find smallest/largest sums
    freqs = np.sort([item['frequency'] for item in cipher_frequencies])
    
    pruned_candidates = []
    for item in cipher_frequencies:
        current_freq = item['frequency']
        
        # Create a temporary array of other frequencies
        others = freqs[freqs != current_freq]
        
        # Condition 1: Is the frequency too high?
        # Sum with the (n-1) smallest other frequencies
        if len(others) >= target_homophones - 1:
            sum_with_smallest = current_freq + np.sum(others[:target_homophones - 1])
            if sum_with_smallest > upper_bound:
                continue # Prune this item

        # Condition 2: Is the frequency too low?
        # Sum with the (n-1) largest other frequencies
        if len(others) >= target_homophones - 1:
            sum_with_largest = current_freq + np.sum(others[-(target_homophones - 1):])
            if sum_with_largest < lower_bound:
                continue # Prune this item
        
        pruned_candidates.append(item)
    
    pruned_candidates.sort(key=lambda x: x['frequency'])
    
    return np.array(pruned_candidates, dtype=cipher_frequencies.dtype)