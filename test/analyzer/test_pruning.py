import numpy as np

from src.analyzer.pruning import prune_frequencies

def test_prune_frequencies():
    """
    Test the pruning of frequencies based on target homophones and bounds.
    """
    cipher_frequencies = np.array([
        {"symbol": "A", "frequency": 0.1},
        {"symbol": "B", "frequency": 0.2},
        {"symbol": "C", "frequency": 0.3},
        {"symbol": "D", "frequency": 0.4},
        {"symbol": "E", "frequency": 0.5},
    ])

    pruned = prune_frequencies(cipher_frequencies, target_homophones=3, lower_bound=0.6, upper_bound=1.0)
    expected_symbols = {"A", "B", "C", "D", "E"}
    pruned_symbols = {item["symbol"] for item in pruned}
    assert pruned_symbols == expected_symbols

def test_prune_frequencies_strict_bounds():
    """
    Test the pruning with strict bounds that eliminate some frequencies.
    """
    cipher_frequencies = np.array([
        {"symbol": "A", "frequency": 0.1},
        {"symbol": "B", "frequency": 0.2},
        {"symbol": "C", "frequency": 0.3},
        {"symbol": "D", "frequency": 0.4},
        {"symbol": "E", "frequency": 0.5},
    ])

    pruned = prune_frequencies(cipher_frequencies, target_homophones=2, lower_bound=0.7, upper_bound=0.9)
    expected_symbols = {"B", "C", "D", "E"}
    pruned_symbols = {item["symbol"] for item in pruned}
    assert pruned_symbols == expected_symbols

def test_less_frequencies_than_homophones():
    """
    Test the case where there are fewer frequencies than target homophones.
    """
    cipher_frequencies = np.array([
        {"symbol": "A", "frequency": 0.1},
        {"symbol": "B", "frequency": 0.2},
    ])

    pruned = prune_frequencies(cipher_frequencies, target_homophones=3, lower_bound=0.1, upper_bound=0.5)
    assert len(pruned) == 0
