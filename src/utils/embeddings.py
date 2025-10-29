import numpy as np
import os
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")


# --- Your Original CSV Parser (Unchanged) ---
# This function is specific to your CSV format, which is fine.
# It handles skipping the known header lines.
def parse_csv(filepath: str) -> list[list[str]]:
	"""Load a CSV file and return its contents as a list of rows.

	Args:
	    filepath (str): Path to the CSV file.

	Returns:
	    list[list[str]]: A list of rows, where each row is a list of strings.
	"""
	with open(filepath) as f:
		parts = []
		for line in f:
			# Skip first line if it contains headers
			if line.startswith("cipher_symbol") or line.startswith("plaintext_letter"):
				continue
			line = line.strip()
			if line:  # Added a check to skip potential empty lines
				parts.append(line.split(","))
		return parts


def load_data_as_dict(
	relative_filepath: str, value_processor: Callable[[List[str]], T]
) -> Dict[str, T]:
	"""
	A general helper to load data from a CSV in the data directory and
	process it into a dictionary.

	It assumes the first column of the CSV is the key.

	Args:
	    relative_filepath (str): The filename, relative to the ../../data dir.
	    value_processor (Callable): A function that takes a row (list of strings)
	                                and returns the processed value for the dict.

	Returns:
	    Dict[str, T]: The processed dictionary.
	"""
	filepath = os.path.join(os.path.dirname(__file__), "../../data", relative_filepath)

	data = parse_csv(filepath)

	processed_dict = {}
	for parts in data:
		if not parts:
			continue
		key = parts[0]
		value: T = value_processor(parts)
		processed_dict[key] = value

	return processed_dict

def get_average_embedding(embeddings: np.ndarray) -> np.ndarray:
	"""Calculate the average embedding vector from a dictionary of embeddings.

	Args:
		embeddings (np.ndarray): A list of embedding vectors.

	Returns:
		np.ndarray: The average embedding vector.
	"""
	return np.mean(embeddings, axis=0)


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


def solve_procrustes(X, Y):
	"""
	Solves the Orthogonal Procrustes problem.
	Finds the optimal rotation matrix W that maps X to Y
	(i.e., minimizes ||XW - Y||^2).

	Args:
		X (np.ndarray): A k x d matrix of k anchor vectors from the source space.
		Y (np.ndarray): A k x d matrix of k anchor vectors from the target space.
						(Must be in the same order as X).

	Returns:
		np.ndarray: The optimal d x d rotation matrix W.
	"""
	# 1. Calculate the matrix M = X.T @ Y
	# This (d x d) matrix captures the covariance between the two sets of points.
	M = X.T @ Y

	# 2. Compute the Singular Value Decomposition (SVD) of M
	# M = U * S * V.T
	U, S, Vt = np.linalg.svd(M)

	# 3. Calculate the optimal rotation matrix W = U @ Vt
	# This is the Procrustes solution. It's the matrix that best
	# aligns X with Y.
	W = U @ Vt

	return W


def build_anchor_matrices(model1_vecs, model2_vecs, partial_key):
	"""
	Builds the X and Y anchor matrices from key-value models.

	Args:
		model1_vecs (dict): Dict of {key: vector} for the source model.
		model2_vecs (dict): Dict of {key: vector} for the target model.
		partial_key (list): A list of tuples [(key1_src, key2_tgt), ...].

	Returns:
		(np.ndarray, np.ndarray): X_anchors (k x d) and Y_anchors (k x d)
	"""
	print(type(model1_vecs))

	X_list = []
	Y_list = []

	for src_key, tgt_key in partial_key:
		if src_key in model1_vecs and tgt_key in model2_vecs:
			X_list.append(model1_vecs[src_key])
			Y_list.append(model2_vecs[tgt_key])
		else:
			print(f"Warning: Key pair ({src_key}, {tgt_key}) not found in models.")

	# Convert lists to k x d matrices
	X_anchors = np.array(X_list)
	Y_anchors = np.array(Y_list)

	if X_anchors.shape[0] == 0:
		raise ValueError("No valid anchor points found in partial key.")

	return X_anchors, Y_anchors


def normalize_vectors(vector_dict):
	"""
	Normalizes all vectors in a dictionary to unit length (L2 norm).

	Args:
		vector_dict (dict): The {key: vector} space to normalize.

	Returns:
		dict: A new dict with the same keys and unit-length vectors.
	"""
	normalized_dict = {}
	for key, vec in vector_dict.items():
		norm = np.linalg.norm(vec)
		if norm > 0:
			normalized_dict[key] = vec / norm
		else:
			normalized_dict[key] = vec  # Avoid division by zero
	return normalized_dict


def find_closest(target_vec, vector_dict):
	"""
	Finds the key in vector_dict with the vector closest to target_vec.
	Assumes all vectors (target_vec and in vector_dict) are normalized.
	Uses dot product (which is equivalent to cosine similarity here).

	Args:
		target_vec (np.ndarray): The 1 x d normalized vector to match.
		vector_dict (dict): The {key: normalized_vector} space to search in.

	Returns:
		(str, float): The key of the closest vector and the similarity score.
	"""
	max_sim = -float("inf")
	closest_key = None

	for key, vec in vector_dict.items():
		# Calculate dot product (cosine similarity for normalized vectors)
		sim = cosine_sim(target_vec, vec)

		if sim > max_sim:
			max_sim = sim
			closest_key = key

	return closest_key, max_sim
