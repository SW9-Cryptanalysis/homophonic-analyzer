from ..load_cipher import load_cipher, get_cipher_frequencies
from ..utils.constants import EXAMPLE_CIPHERS_PATH
from ..utils.utils import load_letter_frequencies
from .feasibility import calculate_target_range, is_feasible
from .backtracking import backtracking


def find_letter_candidates(
	cipher_file_path: str = "c_400_5.json",
) -> list[tuple[str, list[set[int]]]]:
	"""Find candidate sets of cipher symbols for each letter based on frequency.

	Args:
		cipher_file_path (str): Path to the cipher JSON file.

	Returns:
		List[Tuple[str, List[Set[int]]]]: List of tuples where each tuple contains a
				letter and a list of candidate sets of cipher symbols.

	"""
	candidates = []

	cipher = load_cipher(EXAMPLE_CIPHERS_PATH / cipher_file_path)
	cipher_frequencies = get_cipher_frequencies(cipher)
	letter_frequencies = load_letter_frequencies("english")

	sorted_letters = sorted(letter_frequencies.items(), key=lambda item: item[1])

	for letter, freq in sorted_letters:
		min_max_range = calculate_target_range(len(cipher_frequencies), freq)

		is_letter_feasible = is_feasible(len(cipher_frequencies), min_max_range[1])
		if not is_letter_feasible:
			continue

		# Find all subsets of cipher symbols whose frequencies sum to the target range
		letter_candidates = []
		for i in range(min_max_range[0], min_max_range[1] + 1):
			letter_candidates += backtracking(
				cipher_frequencies,
				i,
				freq,
			)

		if letter_candidates:
			candidates.append((letter, letter_candidates))

	return candidates
