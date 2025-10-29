import logging
import numpy as np
from ..utils.embeddings import (
	get_embeddings,
	get_mappings,
	get_average_embedding,
	cosine_sim,
)
from ..utils.constants import (
	EMBEDDINGS_PATH,
	SPECIAL_CIPHER_PREFIX,
	MONOALPHABETIC_CIPHER_PREFIX,
	DATA_PATH,
)
from ..utils.evaluation import ser

def test_embeddings(logger: logging.Logger) -> None:
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
	logger.info(f"Embeddings length: {len(embeddings_30)}")
	logger.info(f"Mappings length: {len(mappings_30)}")

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

	logger.info("Cosine Similarities for each letter:")
	logger.info("------------------------------")
	logger.info("| Letter".center(7) + " | " + "Cosine Similarity".rjust(14) + " |")
	logger.info("------------------------------")
	for letter, sim in cosine_similarities.items():
		logger.info(f"|{letter.center(7)} | {float(sim):>17.4f} |")
	logger.info("------------------------------")
	logger.info(f"Average cosine similarity: {avg_cosine_sim:.4f}")


def test_mono_embeddings(logger: logging.Logger) -> None:
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

	for symbol, letter in sorted(set(mappings.items()), key=lambda x: x[1]):
		similarity = cosine_sim(
			cipher_embeddings[symbol.lower()],
			english_embeddings[letter.lower()],
		)
		logger.info(
			f"  {letter} <-> {symbol:2}, Cosine Similarity: {similarity:7.4f}",
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
			logger.info(
				f"Closest letter to {symbol}: {closest[0]}, Similarity: "
				f"{closest[1]:.4f}",
			)
			reconstructed_key[symbol] = closest[0]
	# Log the reconstructed key
	logger.info("Reconstructed Key:")
	for symbol, letter in reconstructed_key.items():
		logger.info(f"{symbol} -> {letter}")

	with open(DATA_PATH / "example_ciphers" / "example_cipher_text.txt") as f:
		cipher_text = f.read().strip()

	plain_text = []
	for cipher_symbol in cipher_text.split(" "):
		plain_symbol = reconstructed_key.get(cipher_symbol, cipher_symbol)
		plain_text.append(plain_symbol)
	logger.info("Decrypted text: %s", " ".join(plain_text))
	logger.info("SER: %.2f", ser(cipher_text, " ".join(plain_text)))
