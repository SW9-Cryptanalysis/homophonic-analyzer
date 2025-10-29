from typing import Any
from collections.abc import Callable
import numpy as np


def hill_climbing(
	initial_solution: Any,
	get_neighbors: Callable[[Any, float], list[Any]],
	score_function: Callable[[Any], float],
	cfg: dict[str, Any],
) -> tuple[Any, float]:  # type: ignore
	"""Perform hill climbing to find the best solution.

	:param initial_solution: The starting point for the algorithm.
	:param get_neighbors: A function that takes a solution and a step size, and
		returns a list of neighboring solutions.
		I.e solutions that are similar to the current one but with small changes.
	:param score_function: A function that takes a solution and returns its score
		(higher is better).
	:param cfg: A dictionary containing configuration parameters such as
		'max_iterations' and 'step_size'.
	:return: A tuple containing the best solution found and its score.
	"""
	max_iterations = cfg.get("max_iterations", float("inf"))
	step_size = cfg.get("step_size", 0.1)

	current_solution = np.array([initial_solution])
	current_score = score_function(current_solution)

	for _ in range(max_iterations):
		neighbors = get_neighbors(current_solution, step_size)
		if not neighbors:
			break  # No neighbors to explore

		# Evaluate all neighbors and find the best one
		best_neighbor = None
		best_score = current_score

		best_neighbor, best_score = evaluate_neighbors(
			neighbors,
			score_function,
			best_score,
		)

		# If no better neighbor is found, we have reached a local maximum
		if best_neighbor is None:
			break

		# Move to the best neighbor
		current_solution = best_neighbor
		current_score = best_score

	return current_solution, current_score


def evaluate_neighbors(
	neighbors: list[Any],
	score_function: Callable[[Any], float],
	best_score: float,
) -> tuple[Any, float]:
	"""Evaluate a list of neighbors and return the best one along with its score.

	:param neighbors: A list of neighboring solutions to evaluate.
	:param score_function: A function that takes a solution and returns its score
		(higher is better).
	:return: A tuple containing the best neighbor found and its score.
	"""
	best_neighbor = None

	for neighbor in neighbors:
		neighbor_score = score_function(neighbor)
		if neighbor_score > best_score:
			best_neighbor = neighbor
			best_score = neighbor_score

	return best_neighbor, best_score
