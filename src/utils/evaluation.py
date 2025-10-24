def ser(reference: str, hypothesis: str) -> float:
	"""Compute the Symbol Error Rate (SER) between the reference and hypothesis strings."""
	ref_symbols = reference.split(" ")
	hyp_symbols = hypothesis.split(" ")
	errors = sum(1 for r, h in zip(ref_symbols, hyp_symbols) if r != h)
	return errors / len(ref_symbols) if ref_symbols else 0.0
