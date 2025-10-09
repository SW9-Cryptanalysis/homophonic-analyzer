import numpy as np

from src.analyzer.backtracking import backtracking

def test_backtracking_basic():
    """Basic test with small, manageable data"""
    cipher_frequencies = np.array([(1, 0.5), (2, 0.3), (3, 0.2)], dtype=[("symbol", int), ("frequency", float)])
    target_homophones = 2
    target_frequency = 0.8

    results = backtracking(cipher_frequencies, target_homophones, target_frequency)

    expected = [{1, 2}]

    assert len(results) == len(expected)
    for res in results:
        assert res in expected

def test_backtracking_pruning_too_many_symbols():
    """Test that covers line 27: pruning when len(current_combination) > target_homophones

    This condition can only be reached if target_homophones is negative
    """
    cipher_frequencies = np.array([(1, 0.1)], dtype=[("symbol", int), ("frequency", float)])
    target_homophones = -1
    target_frequency = 0.0

    results = backtracking(cipher_frequencies, target_homophones, target_frequency)

    assert len(results) == 0

def test_backtracking_pruning_sum_exceeds_upper_bound():
    """Test that covers line 31: pruning when current_sum > upper_bound

    This condition can be reached when individual frequencies are large and
    target_homophones is bigger than 1.
    """
    cipher_frequencies = np.array([(1, 0.8), (2, 0.9), (3, 0.1)], dtype=[("symbol", int), ("frequency", float)])
    target_homophones = 2
    target_frequency = 0.1

    results = backtracking(cipher_frequencies, target_homophones, target_frequency)

    assert len(results) == 0
