from src.utils.utils import load_letter_frequencies
import pytest
import numpy as np

def test_load_letter_frequencies_structure():
    """Tests the output structure of load_letter_frequencies."""
    frequencies = load_letter_frequencies("english")

    assert isinstance(frequencies, dict)
    assert all(isinstance(key, str) for key in frequencies)
    assert all(isinstance(value, float) for value in frequencies.values())
    assert "E" in frequencies

def test_load_letter_frequencies_sum():
    """Tests that the sum of all frequencies is approximately 1.0."""
    frequencies = load_letter_frequencies("english")

    total_frequency = sum(frequencies.values())

    np.testing.assert_almost_equal(total_frequency, 1.0, decimal=0, err_msg="Sum of frequencies should be close to 1.0")

def test_load_letter_frequencies_case_insensitive():
    """Tests that the language parameter is case-insensitive."""
    frequencies_lower = load_letter_frequencies("english")
    frequencies_upper = load_letter_frequencies("ENGLISH")

    assert frequencies_lower == frequencies_upper

def test_unsupported_language_raises_error():
    """Tests that an unsupported language raises a ValueError."""
    with pytest.raises(ValueError, match="Unsupported language."):
        load_letter_frequencies("german")
