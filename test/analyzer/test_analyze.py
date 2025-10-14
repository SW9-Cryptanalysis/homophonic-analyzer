import numpy as np
from unittest.mock import patch

from src.analyzer.analyze import find_letter_candidates

def test_find_letter_candidates_basic():
    """Basic functionality test with a known cipher file."""
    try:
        candidates = find_letter_candidates("test_cipher.json")
    except Exception as e:
        assert False, f"find_letter_candidates raised an exception: {e}" # noqa: B011

    assert isinstance(candidates, list)
    for letter, candidate_sets in candidates:
        assert isinstance(letter, str) and len(letter) == 1
        assert isinstance(candidate_sets, list)
        for candidate_set in candidate_sets:
            assert isinstance(candidate_set, set)
            for symbol in candidate_set:
                assert isinstance(symbol, np.integer)

def test_performance_with_infeasible_cipher():
    """Test that the algorithm hits feasibility checks and avoids heavy computation."""
    with patch("src.analyzer.feasibility.FEASIBILITY_THRESHOLD", 100_000):
        try:
            candidates = find_letter_candidates("test_cipher.json")
        except Exception as e:
            assert False, f"find_letter_candidates raised an exception: {e}" # noqa: B011

    assert isinstance(candidates, list)
    for letter, candidate_sets in candidates:
        assert isinstance(letter, str) and len(letter) == 1
        assert isinstance(candidate_sets, list)
        for candidate_set in candidate_sets:
            assert isinstance(candidate_set, set)
            for symbol in candidate_set:
                assert isinstance(symbol, np.integer)

def test_find_letter_candidates_perfomance():
    """Performance test to ensure the function runs within a reasonable time frame."""
    import time
    start_time = time.time()
    candidates = find_letter_candidates("test_cipher.json")
    end_time = time.time()
    duration = end_time - start_time
    assert duration < 10, f"find_letter_candidates took too long: {duration} seconds"

    assert isinstance(candidates, list)
    for letter, candidate_sets in candidates:
        assert isinstance(letter, str) and len(letter) == 1
        assert isinstance(candidate_sets, list)
        for candidate_set in candidate_sets:
            assert isinstance(candidate_set, set)
            for symbol in candidate_set:
                assert isinstance(symbol, np.integer)
