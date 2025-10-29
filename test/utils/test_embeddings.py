from src.utils.embeddings import (
	load_data_as_dict,
	get_average_embedding,
	cosine_sim,
	solve_procrustes,
	build_anchor_matrices,
	normalize_vectors,
	find_closest,
)
import os
import numpy as np
import pytest

CURRENT_DIR = os.path.dirname(__file__)


class TestGetEmbeddings:
	@pytest.fixture(scope="class")
	def embeddings(self) -> dict[str, np.ndarray]:
		"""Setup that runs before each test."""
		embeddings = load_data_as_dict(
			os.path.join(CURRENT_DIR, "..", "data", "test_embeddings.csv"),
			lambda parts: np.array([float(x) for x in parts[1:]]),
		)
		return embeddings

	def test_get_embeddings_shape(self, embeddings):
		"""Test that the embeddings have the correct shape."""
		assert isinstance(embeddings, dict)
		for vector in embeddings.values():
			assert isinstance(vector, np.ndarray)
			assert vector.shape == (20,)

	def test_get_embeddings_content(self, embeddings):
		"""Test that the embeddings are loaded correctly."""
		assert isinstance(embeddings, dict)
		for key, vector in embeddings.items():
			assert isinstance(key, str)
			assert isinstance(vector, np.ndarray)
			assert all(isinstance(x, float) for x in vector)

	def test_get_embeddings_empty_parts(self, mocker):
		"""Test that empty parts are handled correctly."""
		mocker.patch(
			"src.utils.embeddings.parse_csv",
			return_value=[[], ["A", "1.0", "2.0"], [], ["B", "3.0", "4.0"]],
		)
		embeddings = load_data_as_dict(
			"dummy_path.csv",
			lambda parts: np.array([float(x) for x in parts[1:]]),
		)
		assert isinstance(embeddings, dict)
		assert len(embeddings) == 2  # Only two valid entries


class TestGetMappings:
	def test_get_mappings_content(self):
		"""Test that the mappings are loaded correctly."""
		mappings = load_data_as_dict(
			os.path.join(CURRENT_DIR, "..", "data", "test_mappings.csv"),
			lambda parts: parts[1],
		)
		assert isinstance(mappings, dict)
		for symbol, letter in mappings.items():
			assert isinstance(symbol, str)
			assert isinstance(letter, str)
			assert len(letter) == 1  # Ensure it's a single letter


class TestGetAverageEmbedding:
	def test_get_average_embedding_value(self):
		"""Test that the average embedding is computed correctly."""
		embeddings = {
			"A": np.array([1.0, 2.0, 3.0]),
			"B": np.array([4.0, 5.0, 6.0]),
			"C": np.array([7.0, 8.0, 9.0]),
		}
		avg_embedding = get_average_embedding(np.array(list(embeddings.values())))
		expected = np.array([4.0, 5.0, 6.0])
		np.testing.assert_array_equal(avg_embedding, expected)


class TestCosineSim:
	def test_cosine_sim_value(self):
		"""Test that the cosine similarity is computed correctly."""
		vec1 = np.array([1.0, 0.0, 0.0])
		vec2 = np.array([0.0, 1.0, 0.0])
		similarity = cosine_sim(vec1, vec2)
		expected = 0.0  # Orthogonal vectors
		assert np.isclose(similarity, expected)

	def test_cosine_sim_identical_vectors(self):
		"""Test that the cosine similarity of identical vectors is 1."""
		vec = np.array([1.0, 2.0, 3.0])
		similarity = cosine_sim(vec, vec)
		expected = 1.0
		assert np.isclose(similarity, expected)

	def test_cosine_sim_opposite_vectors(self):
		"""Test that the cosine similarity of opposite vectors is -1."""
		vec1 = np.array([1.0, 0.0, 0.0])
		vec2 = np.array([-1.0, 0.0, 0.0])
		similarity = cosine_sim(vec1, vec2)
		expected = -1.0
		assert np.isclose(similarity, expected)

	def test_cosine_sim_zero_vector(self):
		"""Test that the cosine similarity involving a zero vector raises an error."""
		vec1 = np.array([0.0, 0.0, 0.0])
		vec2 = np.array([1.0, 2.0, 3.0])
		result = cosine_sim(vec1, vec2)
		assert result == 0.0

	def test_cosine_sim_non_normalized_vectors(self):
		"""Test that the cosine similarity works for non-normalized vectors."""
		vec1 = np.array([2.0, 0.0, 0.0])
		vec2 = np.array([4.0, 0.0, 0.0])
		similarity = cosine_sim(vec1, vec2)
		expected = 1.0  # Same direction
		assert np.isclose(similarity, expected)


class TestSolveProcrustes:
	def test_solve_procrustes_identity(self):
		"""Test that the Procrustes solver returns identity for identical matrices."""
		X = np.array([[1, 0], [0, 1]])
		Y = np.array([[1, 0], [0, 1]])
		W = solve_procrustes(X, Y)
		expected = np.eye(2)
		np.testing.assert_array_almost_equal(W, expected)

	def test_solve_procrustes_rotation(self):
		"""Test that the Procrustes solver returns the correct rotation matrix."""
		theta = np.pi / 4  # 45 degrees
		W_true = np.array(
			[[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
		)
		X = np.array([[1, 0], [0, 1]])
		Y = X @ W_true
		W_estimated = solve_procrustes(X, Y)
		np.testing.assert_array_almost_equal(W_estimated, W_true)


class TestBuildAnchorMatrices:
	def test_build_anchor_matrices_shapes(self):
		"""Test that the anchor matrices have the correct shapes."""
		model1_vecs = {
			"a": np.array([1.0, 0.0]),
			"b": np.array([0.0, 1.0]),
		}
		model2_vecs = {
			"1": np.array([0.0, 1.0]),
			"2": np.array([1.0, 0.0]),
		}
		partial_key = [("a", "2"), ("b", "1")]
		X, Y = build_anchor_matrices(model1_vecs, model2_vecs, partial_key)
		assert X.shape == (2, 2)
		assert Y.shape == (2, 2)

	def test_build_anchor_matrices_content(self):
		"""Test that the anchor matrices are built correctly."""
		model1_vecs = {
			"a": np.array([1.0, 0.0]),
			"b": np.array([0.0, 1.0]),
		}
		model2_vecs = {
			"1": np.array([0.0, 1.0]),
			"2": np.array([1.0, 0.0]),
		}
		partial_key = [("a", "2"), ("b", "1")]
		X, Y = build_anchor_matrices(model1_vecs, model2_vecs, partial_key)
		expected_X = np.array([[1.0, 0.0], [0.0, 1.0]])
		expected_Y = np.array([[1.0, 0.0], [0.0, 1.0]])
		np.testing.assert_array_equal(X, expected_X)
		np.testing.assert_array_equal(Y, expected_Y)

	def test_build_anchor_matrices_missing_key(self):
		"""Test that missing keys raise an error."""
		model1_vecs = {
			"a": np.array([1.0, 0.0]),
		}
		model2_vecs = {
			"1": np.array([0.0, 1.0]),
		}
		partial_key = [("a", "2")]  # "2" is missing in model2_vecs
		with pytest.raises(ValueError):
			build_anchor_matrices(model1_vecs, model2_vecs, partial_key)

	def test_build_anchor_matrices_no_valid_anchors(self):
		"""Test that no valid anchors raise an error."""
		model1_vecs = {
			"a": np.array([1.0, 0.0]),
		}
		model2_vecs = {
			"1": np.array([0.0, 1.0]),
		}
		partial_key = [("b", "2")]  # Both keys are missing
		with pytest.raises(ValueError):
			build_anchor_matrices(model1_vecs, model2_vecs, partial_key)


class TestNormalizeVectors:
	def test_normalize_vectors_unit_length(self):
		"""Test that vectors are normalized to unit length."""
		vector_dict = {
			"a": np.array([3.0, 4.0]),
			"b": np.array([1.0, 0.0]),
		}
		normalized = normalize_vectors(vector_dict)
		for vec in normalized.values():
			norm = np.linalg.norm(vec)
			assert np.isclose(norm, 1.0)

	def test_normalize_vectors_zero_vector(self):
		"""Test that zero vectors remain unchanged."""
		vector_dict = {
			"a": np.array([0.0, 0.0]),
			"b": np.array([1.0, 2.0]),
		}
		normalized = normalize_vectors(vector_dict)
		assert np.array_equal(normalized["a"], np.array([0.0, 0.0]))
		norm_b = np.linalg.norm(normalized["b"])
		assert np.isclose(norm_b, 1.0)

class TestFindClosest:
	def test_find_closest(self):
		"""Test that the closest vector is found correctly."""
		target_vec = np.array([1.0, 0.0])
		vector_dict = {
			"a": np.array([0.9, 0.1]),
			"b": np.array([0.0, 1.0]),
			"c": np.array([1.0, 0.2]),
		}
		closest_key = find_closest(target_vec, vector_dict)
		assert closest_key[0] == "a"
		assert closest_key[1] > 0.99  # High similarity

	def test_find_closest_empty_dict(self):
		"""Test that an empty dictionary returns None."""
		target_vec = np.array([1.0, 0.0])
		vector_dict = {}
		closest_key = find_closest(target_vec, vector_dict)
		assert closest_key[0] is None
		assert closest_key[1] == -float("inf")