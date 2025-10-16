import numpy as np
import pytest
from src.hill_climbing.hill_climbing import hill_climbing


@pytest.fixture
def objective():
	return lambda x: -(x[0] ** 2) + 5  # simple quadratic function with a maximum at x=0


@pytest.fixture
def generate_neighbors():
	def _generate_neighbors(x, step_size=0.1):
		return [np.array([x[0] + step_size]), np.array([x[0] - step_size])]

	return _generate_neighbors


def test_hill_climbing(generate_neighbors, objective):
	initial_solution = 2.0
	solution, value = hill_climbing(
		initial_solution,
		get_neighbors=generate_neighbors,
		score_function=objective,
		cfg={"max_iterations": 100},
	)

	assert np.isclose(solution[0], 0.0, atol=0.1)
	assert np.isclose(value, 5.0, atol=0.1)


def test_no_improvement(generate_neighbors, objective):
	initial_solution = 0.0  # Start at the maximum
	solution, value = hill_climbing(
		initial_solution,
		get_neighbors=generate_neighbors,
		score_function=objective,
		cfg={"max_iterations": 100},
	)
	assert solution[0] == 0.0  # at maximum
	assert value == 5.0  # maximum value


def test_no_neighbors(objective):
	initial_solution = 2.0

	def no_neighbors(x, step_size=0.1):
		return [] 

	solution, value = hill_climbing(
		initial_solution,
		get_neighbors=no_neighbors,
		score_function=objective,
		cfg={"max_iterations": 100},
	)
	assert solution[0] == initial_solution 
	assert value == objective(
		np.array([initial_solution])
	)
