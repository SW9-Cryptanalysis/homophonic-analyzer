from src.utils.evaluation import ser

def test_ser_perfect_match():
	"""Test SER when reference and hypothesis are identical."""
	reference = "A B C D E"
	hypothesis = "A B C D E"
	expected_ser = 0.0

	computed_ser = ser(reference, hypothesis)

	assert computed_ser == expected_ser, f"Expected SER: {expected_ser}, but got: {computed_ser}"
 
def test_ser_all_symbols_different():
	"""Test SER when all symbols are different."""
	reference = "A B C D E"
	hypothesis = "F G H I J"
	expected_ser = 1.0

	computed_ser = ser(reference, hypothesis)

	assert computed_ser == expected_ser, f"Expected SER: {expected_ser}, but got: {computed_ser}"
 
def test_ser_partial_match():
	"""Test SER with partial matches."""
	reference = "A B C D E"
	hypothesis = "A X C Y E"
	expected_ser = 0.4  # 2 errors out of 5 symbols

	computed_ser = ser(reference, hypothesis)

	assert computed_ser == expected_ser, f"Expected SER: {expected_ser}, but got: {computed_ser}"