import numpy as np

from src.analyzer.pruning import prune_frequencies

def test_prune_frequencies():
    """
    Test the pruning of frequencies based on target homophones and bounds.
    """
    cipher_frequencies = np.array([
        {'symbol': 'A', 'frequency': 0.1},
        {'symbol': 'B', 'frequency': 0.2},
        {'symbol': 'C', 'frequency': 0.3},
        {'symbol': 'D', 'frequency': 0.4},
        {'symbol': 'E', 'frequency': 2.0},
    ])
    
    # Test case where some frequencies should be pruned
    pruned = prune_frequencies(cipher_frequencies, target_homophones=3, lower_bound=0.6, upper_bound=1.0)
    pruned_symbols = {item['symbol'] for item in pruned}
    assert pruned_symbols == {'A', 'B', 'C', 'D'}
    
    # Test case where no frequencies can meet the criteria
    pruned = prune_frequencies(cipher_frequencies, target_homophones=4, lower_bound=5.0, upper_bound=7.5)
    assert len(pruned) == 0
    
    # Test case where all frequencies are valid
    pruned = prune_frequencies(cipher_frequencies, target_homophones=2, lower_bound=0.3, upper_bound=0.7)
    pruned_symbols = {item['symbol'] for item in pruned}
    assert pruned_symbols == {'A', 'B', 'C', 'D'}
    
def test_less_frequencies_than_homophones():
    """
    Test the case where there are fewer frequencies than target homophones.
    """
    cipher_frequencies = np.array([
        {'symbol': 'A', 'frequency': 0.1},
        {'symbol': 'B', 'frequency': 0.2},
    ])
    
    pruned = prune_frequencies(cipher_frequencies, target_homophones=3, lower_bound=0.1, upper_bound=0.5)
    assert len(pruned) == 0