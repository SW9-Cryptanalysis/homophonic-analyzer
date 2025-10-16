import numpy as np
import os



def get_embeddings(filepath: str) -> dict[str, np.ndarray]:
	"""Load embeddings from a CSV file.
	Args:
		filepath (str): Path to the CSV file containing embeddings.
	Returns:
		dict: A dictionary mapping symbols to their corresponding embedding vectors.
	"""
	filepath = os.path.join(os.path.dirname(__file__), '../../data', filepath)

	embeddings = {}
	with open(filepath, 'r') as f:
		for line in f:
			# Skip first line if it contains headers
			if line.startswith('cipher_symbol') or line.startswith('plaintext_letter'):
				continue
			parts = line.strip().split(',')
			symbol = parts[0]
			vector = np.array([float(x) for x in parts[1:]])
			embeddings[symbol] = vector
	return embeddings

def get_mappings(filepath: str) -> dict[str, str]:
	"""Load symbol-to-letter mappings from a CSV file.
	Args:
		filepath (str): Path to the CSV file containing mappings.
	Returns:
		dict: A dictionary mapping symbols to their corresponding letters.
	"""
	filepath = os.path.join(os.path.dirname(__file__), '../../data', filepath)

	mappings = {}
	with open(filepath, 'r') as f:
		for line in f:
			# Skip first line if it contains headers
			if line.startswith('cipher_symbol'):
				continue
			parts = line.strip().split(',')
			symbol = parts[0]
			letter = parts[1]
			mappings[symbol] = letter
	return mappings

def get_average_embedding(embeddings: np.ndarray) -> np.ndarray:
	"""Calculate the average embedding vector from a dictionary of embeddings.
	Args:
		embeddings (dict): A dictionary mapping symbols to their corresponding embedding vectors.
	Returns:
		np.ndarray: The average embedding vector.
	"""
	average_vector = np.mean(embeddings, axis=0)
	return average_vector

def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
	"""Calculate the cosine similarity between two vectors.
	Args:
		vec1 (np.ndarray): The first vector.
		vec2 (np.ndarray): The second vector.
	Returns:
		float: The cosine similarity between the two vectors.
	"""
	if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
		return 0.0
	return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))