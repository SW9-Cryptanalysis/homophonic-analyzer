import numpy as np
import os
from collections.abc import Callable
import logging


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


def load_data_as_dict[T](
	relative_filepath: str,
	value_processor: Callable[[list[str]], T],
) -> dict[str, T]:
	"""Load data from a CSV in the data directory and process it into a dictionary.

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


def solve_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	"""Solve the Orthogonal Procrustes problem.

	Finds the optimal rotation matrix w that maps x to y
	(i.e., minimizes ||xw - y||^2).

	Args:
		x (np.ndarray): A k x d matrix of k anchor vectors from the source space.
		y (np.ndarray): A k x d matrix of k anchor vectors from the target space.
						(Must be in the same order as x).

	Returns:
		np.ndarray: The optimal d x d rotation matrix w.

	"""
	# 1. Calculate the matrix m = x.T @ y 											#noqa
	# This (d x d) matrix captures the covariance between the two sets of points. 	#noqa
	m = x.T @ y

	# 2. Compute the Singular Value Decomposition (SVD) of m 						#noqa
	# m = u * s * v_t 																#noqa
	u, s, v_t = np.linalg.svd(m)

	# 3. Calculate the optimal rotation matrix w = u @ Vt 							#noqa
	w = u @ v_t

	return w


def build_anchor_matrices(
	model1_vecs: dict[str, np.ndarray],
	model2_vecs: dict[str, np.ndarray],
	partial_key: list[tuple[str, str]],
	logger: logging.Logger | None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Build the x and y anchor matrices from key-value models.

	Args:
		model1_vecs (dict): Dict of {key: vector} for the source model.
		model2_vecs (dict): Dict of {key: vector} for the target model.
		partial_key (list): A list of tuples [(key1_src, key2_tgt), ...].
		logger (logging.Logger | None): Optional logger for logging warning

	Returns:
		(np.ndarray, np.ndarray): x_anchors (k x d) and y_anchors (k x d)

	"""
	x_list = []
	y_list = []

	for src_key, tgt_key in partial_key:
		if src_key in model1_vecs and tgt_key in model2_vecs:
			x_list.append(model1_vecs[src_key])
			y_list.append(model2_vecs[tgt_key])
		elif logger: # pragma: no cover
			logger.warning(
				f"Warning: Key pair ({src_key}, {tgt_key}) not found in models.",
			)

	# Convert lists to k x d matrices
	x_anchors = np.array(x_list)
	y_anchors = np.array(y_list)

	if x_anchors.shape[0] == 0:
		raise ValueError("No valid anchor points found in partial key.")

	return x_anchors, y_anchors


def normalize_vectors(vector_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
	"""Normalize all vectors in a dictionary to unit length (L2 norm).

	Args:
		vector_dict (dict[str, np.ndarray]): The {key: vector} space to normalize.

	Returns:
		dict[str, np.ndarray]: A new dict with the same keys and unit-length vectors.

	"""
	normalized_dict = {}
	for key, vec in vector_dict.items():
		norm = np.linalg.norm(vec)
		if norm > 0:
			normalized_dict[key] = vec / norm
		else:
			normalized_dict[key] = vec  # Avoid division by zero
	return normalized_dict


def find_closest(
	target_vec: np.ndarray,
	vector_dict: dict[str, np.ndarray],
) -> tuple[str | None, float]:
	"""Find the key in vector_dict with the vector closest to target_vec.

	Assumes all vectors (target_vec and in vector_dict) are normalized.
	Uses dot product (which is equivalent to cosine similarity here).

	Args:
		target_vec (np.ndarray): The 1 x d normalized vector to match.
		vector_dict (dict[str, np.ndarray]): The {key: normalized_vector} space
			to search in.

	Returns:
		(str | None, float): The key of the closest vector and the similarity score.

	"""
	max_sim = -float("inf")
	closest_key = None

	for key, vec in vector_dict.items():
		sim = cosine_sim(target_vec, vec)

		if sim > max_sim:
			max_sim = sim
			closest_key = key

	return closest_key, max_sim
