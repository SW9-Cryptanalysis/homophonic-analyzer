import numpy as np
from src.load_cipher import load_cipher, get_cipher_frequencies
from src.constants import EXAMPLE_CIPHERS_PATH

def test_load_cipher():
    """Tests that the cipher is loaded correctly."""
    cipher = load_cipher(EXAMPLE_CIPHERS_PATH / ("cipher-1.json"))

    assert isinstance(cipher, list)
    assert all(isinstance(x, int) for x in cipher)
    assert len(cipher) > 0

def test_get_cipher_frequencies_structure():
    """Tests the output structure of get_cipher_frequencies."""
    cipher_text = [1, 2, 2, 3, 3, 3]
    frequencies = get_cipher_frequencies(cipher_text)

    assert isinstance(frequencies, np.ndarray)
    assert frequencies.dtype.names == ("symbol", "frequency")
    assert frequencies.dtype["symbol"] == np.dtype(int)
    assert frequencies.dtype["frequency"] == np.dtype(float)

def test_get_cipher_frequencies_calculation():
    """Tests the frequency calculation of get_cipher_frequencies."""
    cipher_text = [10, 20, 10, 30, 10, 20]
    frequencies = get_cipher_frequencies(cipher_text)

    frequencies = np.sort(frequencies, order="symbol")

    expected_symbols = np.array([10, 20, 30])
    expected_frequencies = np.array([3/6, 2/6, 1/6])

    np.testing.assert_array_equal(frequencies["symbol"], expected_symbols)
    np.testing.assert_almost_equal(frequencies["frequency"], expected_frequencies)

def test_integration_sum_of_frequencies():
    """Tests that the sum of all frequencies is approximately 1.0."""
    cipher = load_cipher(EXAMPLE_CIPHERS_PATH / ("cipher-1.json"))

    frequencies = get_cipher_frequencies(cipher)

    total_frequency = np.sum(frequencies["frequency"])
    np.testing.assert_almost_equal(total_frequency, 1.0, 0, "Sum of frequencies should be excactly 1.0")
