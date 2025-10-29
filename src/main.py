import logging
import os
import numpy as np
import dotenv

from .analyzer.analyze import find_letter_candidates
from .hill_climbing.hill_climbing import hill_climbing
from .utils.logging import AnsiColorFormatter
from .experiments.embeddings import test_mono_embeddings

dotenv.load_dotenv()

handler = logging.StreamHandler()
handler.setFormatter(fmt=AnsiColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "CRITICAL"))


def main() -> None:
	"""Demonstrate the usage of find_letter_candidates."""
	res = find_letter_candidates("cipher-1.json")
	for letter, candidates in res:
		logger.info(f"Letter: {letter}:")
		for candidate_set in candidates:
			logger.info(f"  Candidate set: {[int(cand) for cand in candidate_set]}")


def main_hill_climbing() -> None:
	"""Demonstrate the usage of hill_climbing."""
	cipher_symbols = np.array(
		[
			3.0,
			2.0,
			8.0,
			9.0,
			4.0,
			1.0,
			6.0,
			7.0,
			0.0,
			5.0,
			-1.0,
			-2.0,
			-3.0,
			-4.0,
			-5.0,
		],
	)
	cipher_symbols = sorted(cipher_symbols)

	def score_function(x: np.ndarray) -> float:
		return -(x[0] ** 2) + 5

	def get_neighbors(solution: np.ndarray, step_size: float) -> list[np.ndarray]:
		neighbors = []
		# Changing element to the adjacent symbols in cipher_symbols
		for i in range(len(solution)):
			current_value = solution[i]
			if current_value in cipher_symbols:
				idx = cipher_symbols.index(current_value)
				if idx > 0:
					new_solution = solution.copy()
					new_solution[i] = cipher_symbols[idx - 1]
					neighbors.append(new_solution)
				if idx < len(cipher_symbols) - 1:
					new_solution = solution.copy()
					new_solution[i] = cipher_symbols[idx + 1]
					neighbors.append(new_solution)

		return neighbors

	# Pick random initial solution from cipher_symbols
	initial_solution = np.random.choice(cipher_symbols)
	logger.info(f"Initial solution: {initial_solution}")
	cfg = {
		"max_iterations": 100,
		"step_size": 0.1,
	}

	best_solution, best_score = hill_climbing(
		initial_solution,
		get_neighbors,
		score_function,
		cfg,
	)
	logger.info(f"Best solution: {best_solution}, Best score: {best_score}")


if __name__ == "__main__":
	test_mono_embeddings(logger)
