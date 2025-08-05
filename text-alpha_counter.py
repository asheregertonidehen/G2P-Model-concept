import sys
import unicodedata

def extract_alphabet(file_path):
    unique_chars = set()

    # Read file and collect only letters
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for char in line:
                # Keep only letters (Unicode category "L")
                if unicodedata.category(char).startswith('L'):
                    unique_chars.add(char)

    return sorted(unique_chars)

def main():
    if len(sys.argv) != 2:
        print("Usage: python edo_alphabet_extractor.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    unique_chars = extract_alphabet(input_file)

    print("Alphabet in the file (letters only):")
    for char in unique_chars:
        print(f"{char} (U+{ord(char):04X})")

if __name__ == "__main__":
    main()
