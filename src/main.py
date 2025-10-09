from .analyzer.analyze import find_letter_candidates
import logging

logging.basicConfig(level=logging.INFO)

def main() -> None:
    """Demonstrate the usage of find_letter_candidates."""
    res = find_letter_candidates("cipher-1.json")
    for letter, candidates in res:
        logging.info(f"Letter: {letter}:")
        for candidate_set in candidates:
            logging.info(f"  Candidate set: {[int(cand) for cand in candidate_set]}")


if __name__ == "__main__":
    main()
