import numpy as np
import logging
from ..utils.constants import FREQUENCY_TOLERANCE
from .pruning import prune_frequencies

logger = logging.getLogger(__name__)

def backtracking(
    cipher_frequencies: np.ndarray,
    target_homophones: int,
    target_sum: float,
    correct_symbols: set[int] | None = None,
) -> tuple[int, bool]:
    """Count subsets of cipher symbols whose frequencies sum to a target value.

    Includes a pruning pre-processing step to improve performance.

    Args:
        cipher_frequencies: Array of dicts with 'symbol' and 'frequency' keys.
        target_homophones: Number of symbols to select.
        target_sum: Desired sum of the selected frequencies.
        correct_symbols: Set of correct symbols for this letter (for validation).

    Returns:
        Tuple of (total_candidates_found, correct_candidate_found).

    """
    lower_bound = target_sum - FREQUENCY_TOLERANCE
    upper_bound = target_sum + FREQUENCY_TOLERANCE

    pruned_frequencies = prune_frequencies(
        cipher_frequencies, target_homophones, lower_bound, upper_bound,
    )

    if len(pruned_frequencies) < target_homophones:
        return (0, False)

    stats = {'total_candidates': 0, 'correct_found': False}
    _backtrack(
        start_index=0,
        current_combination=set(),
        current_sum=0.0,
        pruned_frequencies=pruned_frequencies,
        target_homophones=target_homophones,
        target_sum=target_sum,
        bounds=(lower_bound, upper_bound),
        correct_symbols=correct_symbols,
        stats=stats,
    )

    logger.info(f'   Target Homophones: {target_homophones}:')
    logger.info(f'          {stats["total_candidates"]}')
    return (stats['total_candidates'], stats['correct_found'])


def _backtrack(
    start_index: int,
    current_combination: set,
    current_sum: float,
    pruned_frequencies: np.ndarray,
    target_homophones: int,
    target_sum: float,
    bounds: tuple,
    correct_symbols: set[int] | None,
    stats: dict,
) -> None:
    """Count combinations of cipher symbols and check for correct match.

    Args:
        start_index: Current index in pruned_frequencies to consider.
        current_combination: Set of currently selected symbols.
        current_sum: Sum of frequencies in the current combination.
        pruned_frequencies: The pruned array of frequency dicts.
        target_homophones: Number of symbols to select.
        target_sum: Desired sum of the selected frequencies.
        bounds: Tuple (lower_bound, upper_bound) for valid sums.
        correct_symbols: Set of correct symbols for validation (optional).
        stats: Dictionary to track total_candidates and correct_found.

    Returns:
        None. Stats are updated in the stats dictionary.

    """
    lower_bound, upper_bound = bounds
    n = len(pruned_frequencies)

    # Base case: Found a combination of the correct size
    if len(current_combination) == target_homophones:
        if lower_bound <= current_sum <= upper_bound:
            stats['total_candidates'] += 1
            # Check if this matches the correct symbols
            if correct_symbols is not None:
                candidate_symbols = {int(s) for s in current_combination}
                if candidate_symbols == correct_symbols:
                    stats['correct_found'] = True
        return

    # Pruning cases
    if start_index >= n or len(current_combination) > target_homophones:
        return

    for i in range(start_index, n):
        symbol = pruned_frequencies[i]["symbol"]
        frequency = pruned_frequencies[i]["frequency"]

        if current_sum + frequency > upper_bound:
            break

        current_combination.add(symbol)
        _backtrack(
            i + 1, current_combination, current_sum + frequency,
            pruned_frequencies, target_homophones, target_sum,
            bounds, correct_symbols, stats,
        )
        current_combination.remove(symbol)
