import numpy as np
from unittest.mock import patch

from src.analyzer.analyze import find_letter_candidates

def test_find_letter_candidates_basic():
    try:
        candidates = find_letter_candidates("test_cipher.json")
    except Exception as e:
        assert False, f"find_letter_candidates raised an exception: {e}"
        
    assert isinstance(candidates, list), "Expected candidates to be a list"
    assert all(isinstance(c, set) for c in candidates), "Each candidate should be a set"
    assert all(all(isinstance(sym, np.integer) for sym in c) for c in candidates), "Each symbol in candidates should be a numpy integer"
    assert len(candidates) > 0, "Expected at least one candidate to be found"

def test_performance_with_infeasible_cipher():
    """Test that the algorithm hits feasibility checks and avoids heavy computation."""
    with patch('src.analyzer.feasibility.FEASIBILITY_THRESHOLD', 100_000):
        try:
            candidates = find_letter_candidates("test_cipher.json")
        except Exception as e:
            assert False, f"find_letter_candidates raised an exception: {e}"
        
    assert isinstance(candidates, list), "Expected candidates to be a list"
    assert all(isinstance(c, set) for c in candidates), "Each candidate should be a set"
    assert all(all(isinstance(sym, np.integer) for sym in c) for c in candidates), "Each symbol in candidates should be a numpy integer"
    assert len(candidates) > 0, "Expected at least one candidate to be found"