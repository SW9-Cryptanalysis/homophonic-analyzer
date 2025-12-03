import logging
import psutil

from ..load_cipher import load_cipher, get_cipher_frequencies, get_cipher_symbols_given_letter
from ..utils.constants import EXAMPLE_CIPHERS_PATH
from ..utils.utils import load_letter_frequencies
from .feasibility import calculate_target_range, is_feasible
from .backtracking import backtracking

logger = logging.getLogger(__name__)


def find_letter_candidates(
	cipher_file_path: str,
) -> list[tuple[str, list[set[int]]]]:
	"""Find candidate sets of cipher symbols for each letter based on frequency.

	Args:
		cipher_file_path (str): Path to the cipher JSON file.

	Returns:
		List[Tuple[str, List[Set[int]]]]: List of tuples where each tuple contains a
				letter and a list of candidate sets of cipher symbols.

	"""
	candidates = []
	total_number_of_candidates = 0
	correct_candidates = 0

	cipher = load_cipher(EXAMPLE_CIPHERS_PATH / cipher_file_path)
	cipher_frequencies = get_cipher_frequencies(cipher)
	letter_frequencies = load_letter_frequencies("english")

	sorted_letters = sorted(letter_frequencies.items(), key=lambda item: item[1], reverse=True)

	for letter, freq in sorted_letters:
		logger.info(f'Letter: {letter}:')
		correct_letter_symbols = get_cipher_symbols_given_letter(cipher_file_path, letter)
		min_max_range = calculate_target_range(len(cipher_frequencies), freq)

		is_letter_feasible = is_feasible(len(cipher_frequencies), min_max_range[1])
		if not is_letter_feasible:
			logger.info(f'Skipping letter "{letter}" as it is not feasible.')
			continue

		# Find all subsets of cipher symbols whose frequencies sum to the target range
		letter_candidates = []
		for i in range(min_max_range[0], min_max_range[1] + 1):
			# backtracking returns type list[set[int]]
			letter_candidates += backtracking(
				cipher_frequencies,
				i,
				freq,
			)
			total_number_of_candidates += len(letter_candidates)

		if letter_candidates:
			# Check if correct symbols are in any candidate set
			for candidate_tuple in letter_candidates:
				# candidate_tuple is (set_of_symbols, distance)
				candidate_set, distance = candidate_tuple
				# Convert numpy types to native Python types for cleaner logging
				candidate_symbols = {int(s) for s in candidate_set}
				
				if correct_letter_symbols == candidate_symbols:
					correct_candidates += 1
			# candidates.append((letter, letter_candidates))
		
		# Log memory usage after processing each letter
		process = psutil.Process()
		memory_mb = process.memory_info().rss / 1024 / 1024
		logger.info(f'  Memory usage after letter "{letter}": {memory_mb:.2f} MB')

	logger.info(f'Total number of correct candidates: {correct_candidates} out of {len(sorted_letters)} letters')
	logger.info(f'Total number of candidates found: {total_number_of_candidates}')
	return candidates
