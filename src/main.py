import logging
import numpy as np

from .analyzer.analyze import find_letter_candidates
from .hill_climbing.hill_climbing import hill_climbing

from .utils.constants import (
	EMBEDDINGS_PATH,
	MONOALPHABETIC_CIPHER_PREFIX,
	SPECIAL_CIPHER_PREFIX,
	DATA_PATH,
)
from .utils.embeddings import (
	get_embeddings,
	get_mappings,
	get_average_embedding,
	cosine_sim,
)
from .utils.evaluation import ser

logging.basicConfig(level=logging.INFO)


def main() -> None:
	"""Demonstrate the usage of find_letter_candidates."""
	res = find_letter_candidates("cipher-1.json")
	for letter, candidates in res:
		logging.info(f"Letter: {letter}:")
		for candidate_set in candidates:
			logging.info(f"  Candidate set: {[int(cand) for cand in candidate_set]}")


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
	logging.info(f"Initial solution: {initial_solution}")
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
	logging.info(f"Best solution: {best_solution}, Best score: {best_score}")


def test_embeddings() -> None:
	"""Test function for embeddings analysis."""
	embeddings_30 = get_embeddings(
		str(EMBEDDINGS_PATH / f"{SPECIAL_CIPHER_PREFIX}cipher_embeddings.csv"),
	)
	mappings_30 = get_mappings(
		str(EMBEDDINGS_PATH / f"{SPECIAL_CIPHER_PREFIX}mappings.csv"),
	)
	plaintext_embeddings = get_embeddings(
		str(EMBEDDINGS_PATH / f"{SPECIAL_CIPHER_PREFIX}plaintext_embeddings.csv"),
	)
	logging.info(f"Embeddings length: {len(embeddings_30)}")
	logging.info(f"Mappings length: {len(mappings_30)}")

	cosine_similarities: dict[str, float] = {}
	for letter in sorted(set(mappings_30.values())):
		mappings = np.array(
			[
				embeddings_30[key]
				for key, mapping in mappings_30.items()
				if mapping == letter and key in embeddings_30
			],
		)
		avg_embedding = get_average_embedding(mappings)
		cosine_similarities[letter] = cosine_sim(
			avg_embedding,
			plaintext_embeddings[letter.lower()],
		)

	avg_cosine_sim = np.mean([float(sim) for sim in cosine_similarities.values()])

	logging.info("Cosine Similarities for each letter:")
	logging.info("------------------------------")
	logging.info("| Letter".center(7) + " | " + "Cosine Similarity".rjust(14) + " |")
	logging.info("------------------------------")
	for letter, sim in cosine_similarities.items():
		logging.info(f"|{letter.center(7)} | {float(sim):>17.4f} |")
	logging.info("------------------------------")
	logging.info(f"Average cosine similarity: {avg_cosine_sim:.4f}")


def test_mono_embeddings() -> None:
	"""Test function for monoalphabetic cipher embeddings analysis."""
	cipher_embeddings = get_embeddings(
		str(EMBEDDINGS_PATH / f"{MONOALPHABETIC_CIPHER_PREFIX}cipher_embeddings.csv"),
	)
	mappings = get_mappings(
		str(EMBEDDINGS_PATH / f"{MONOALPHABETIC_CIPHER_PREFIX}mappings.csv"),
	)
	english_embeddings = get_embeddings(
		str(EMBEDDINGS_PATH / "english_plaintext_embeddings.csv"),
	)

	reconstructed_key = {}

	for symbol, letter in sorted(set(mappings.items())):
		similarity = cosine_sim(
			cipher_embeddings[symbol.lower()],
			english_embeddings[letter.lower()],
		)
		logging.info(
			f"	{letter} <-> {symbol:2}, Cosine Similarity: {similarity:7.4f}",
		)
		closest = None
		for letter in sorted(set(mappings.values())):
			similarity = cosine_sim(
				cipher_embeddings[symbol.lower()],
				english_embeddings[letter.lower()],
			)
			if closest is None or similarity > closest[1]:
				closest = (letter, similarity)
		if closest:
			logging.info(
				f"Closest letter to {symbol}: {closest[0]}, Similarity: "
						"{closest[1]:.4f}",
			)
			reconstructed_key[symbol] = closest[0]
	# Log the reconstructed key
	logging.info("Reconstructed Key:")
	for symbol, letter in reconstructed_key.items():
		logging.info(f"{symbol} -> {letter}")

	with open(DATA_PATH / "example_ciphers" / "example_cipher_text.txt") as f:
		cipher_text = f.read().strip()

	plain_text = []
	for cipher_symbol in cipher_text.split(" "):
		plain_symbol = reconstructed_key.get(cipher_symbol, cipher_symbol)
		plain_text.append(plain_symbol)
	logging.info("Decrypted text: %s", " ".join(plain_text))
	logging.info("SER: %.2f", ser(cipher_text, " ".join(plain_text)))


if __name__ == "__main__":
	test_mono_embeddings()
