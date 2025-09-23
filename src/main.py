from load_cipher import load_cipher, get_cipher_frequencies

def main():
    print("Hello from homophonic-analyzer!")


if __name__ == "__main__":
    cipher = load_cipher("cipher.json")
    frequencies = get_cipher_frequencies(cipher)
    print(frequencies)
