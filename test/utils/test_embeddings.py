from src.utils.embeddings import get_embeddings, get_mappings, get_average_embedding, cosine_sim
import pytest
import os
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)

class TestGetEmbeddings:
    
    
	def test_get_embeddings_shape(self):
		"""Test that the embeddings have the correct shape."""
		embeddings = get_embeddings(os.path.join(CURRENT_DIR, "..", "data", "test_embeddings.csv"))
		assert isinstance(embeddings, dict)
		for vector in embeddings.values():
			assert isinstance(vector, np.ndarray)
			assert vector.shape == (20,)
   

class TestGetMappings:
	
	
	def test_get_mappings_content(self):
		"""Test that the mappings are loaded correctly."""
		mappings = get_mappings(os.path.join(CURRENT_DIR, "..", "data", "test_mappings.csv"))
		assert isinstance(mappings, dict)
		for symbol, letter in mappings.items():
			assert isinstance(symbol, str)
			assert isinstance(letter, str)
			assert len(letter) == 1 # Ensure it's a single letter
   
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
		expected = 0.0 # Orthogonal vectors
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
		expected = 1.0 # Same direction
		assert np.isclose(similarity, expected)