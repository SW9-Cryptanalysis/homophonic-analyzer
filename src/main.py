from .analyzer.analyze import find_letter_candidates


def main():
    res = find_letter_candidates("cipher-1.json")
    for letter, candidates in res:
        print(f"Letter: {letter}:")
        for candidate_set in candidates:
            print(f"  Candidate set: {[int(cand) for cand in candidate_set]}")


if __name__ == "__main__":
    main()
