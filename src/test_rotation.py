import numpy as np
from utils.embeddings import get_embeddings, get_mappings, normalize_vectors, build_anchor_matrices, solve_procrustes, find_closest


# --- Main execution block to demonstrate the attack ---
if __name__ == "__main__":
	# --- 1. Mock Data (Replace with your real data) ---

	# Our "perfect" Generalized Model (the "source")
	generalized_model_vectors = get_embeddings(
		"embeddings/english_plaintext_embeddings.csv"
	)

	generalized_model_vectors = normalize_vectors(generalized_model_vectors)

	# Our "sparse/noisy" Cipher Model (the "target")
	# These vectors are *not* aligned with the generalized model.
	# Note: '?' is the *true* cipher for 'a', but we don't know that.
	cipher_model_vectors = get_embeddings(
		"embeddings/monoalphabetic-cipher_cipher_embeddings.csv"
	)
 
	cipher_model_vectors = normalize_vectors(cipher_model_vectors)

	# Our partial key (the "anchor points")
	partial_key = [("e", "23"), ("t", "12")]

	print(f"--- Starting Alignment Experiment ---")
	print(f"Partial key: {partial_key}\n")

	# --- 2. Build Anchor Matrices ---
	try:
		X_anchors, Y_anchors = build_anchor_matrices(
			generalized_model_vectors, cipher_model_vectors, partial_key
		)
		print(
			f"Built anchor matrices: X shape {X_anchors.shape}, Y shape {Y_anchors.shape}"
		)

		# --- 3. Solve for the Rotation Matrix W ---
		W = solve_procrustes(X_anchors, Y_anchors)
		print(f"\nSolved for rotation matrix W (shape {W.shape}).")

		# --- 4. Define Keys to Attack ---
		known_source_symbols = {key for key, val in partial_key}
		unknown_source_keys = [
			key for key in generalized_model_vectors.keys()
			if key not in known_source_symbols
		]
		print(f"Unknown source keys to find: {unknown_source_keys}")

		# --- 5. Define Search Space of Unknown Cipher Symbols ---
		known_cipher_symbols = {val for key, val in partial_key}
		search_space = {
			key: vec for key, vec in cipher_model_vectors.items() 
			if key not in known_cipher_symbols
		}
		print(f"Unknown cipher symbols to match against: {list(search_space.keys())}\n")
		
		# --- 6. Loop and Generate Full Key ---
		
		predicted_key_map = {}
		
		print("--- Running Attack Loop ---")
		for source_key in unknown_source_keys:
			if not search_space:
				print("Warning: No more cipher symbols left to match.")
				break
				
			# Get the source vector
			source_vec = generalized_model_vectors[source_key]
			
			# Apply the Rotation
			rotated_vec = source_vec @ W
			rotated_vec /= np.linalg.norm(rotated_vec) # Normalize
			
			# Find the best match *from the remaining symbols*
			predicted_cipher_key, similarity = find_closest(
				rotated_vec,
				search_space
			)
			
			# Store the mapping
			predicted_key_map[source_key] = predicted_cipher_key
			
			print(f"Plaintext '{source_key}' -> Cipher '{predicted_cipher_key}' (Sim: {similarity:.4f})")
			
			# Remove the matched key from the search space to ensure 1-to-1
			del search_space[predicted_cipher_key]
			

		print("\n--- FINAL PREDICTED KEY ---")
		# Add the known partial key to the map for a full view
		for src, tgt in partial_key:
			predicted_key_map[src] = tgt
			
		print(sorted(predicted_key_map.items(), key=lambda x: x[0].lower()))
		
		# Check against our mock ground truth
		ground_truth = {v.lower(): k for k, v in get_mappings(
			"embeddings/monoalphabetic-cipher_mappings.csv"
		).items()}

		print("\n--- GROUND TRUTH KEY ---")
		print(sorted(ground_truth.items(), key=lambda x: x[0].lower()))
		correct_count = 0
		for key, val in predicted_key_map.items():
			if ground_truth.get(key) == val:
				correct_count += 1
		
		print(f"\nAccuracy: {correct_count / len(ground_truth):.0%}")
		print(f"Correct mappings: {correct_count} out of {len(ground_truth)}")

	except Exception as e:
		print(f"\nAn error occurred: {e}")
		print("This can happen if the partial key is empty or not found.")
