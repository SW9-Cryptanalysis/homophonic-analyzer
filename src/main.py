from load_cipher import load_cipher, get_cipher_frequencies
from utils import load_letter_frequencies

def main():
    cipher = load_cipher("cipher.json")
    frequencies = get_cipher_frequencies(cipher)
    print(frequencies)
    frequencies = load_letter_frequencies()
    print(frequencies)


if __name__ == "__main__":
    main()
