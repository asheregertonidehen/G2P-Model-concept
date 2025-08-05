import sys
import unicodedata
import re

def extract_alphabet_and_words(file_path):
    unique_chars = set()
    unique_words = set()

    # Regex to extract words (unicode letters only)
    word_pattern = re.compile(r'\b[^\W\d_]+\b', re.UNICODE)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Collect unique chars that are letters
            for char in line:
                if unicodedata.category(char).startswith('L'):
                    unique_chars.add(char)
            
            # Extract words (tokenize)
            words = word_pattern.findall(line.lower())  # lowercase for uniqueness
            unique_words.update(words)

    return sorted(unique_chars), unique_words

def main():
    if len(sys.argv) != 2:
        print("Usage: python edo_alphabet_extractor.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    unique_chars, unique_words = extract_alphabet_and_words(input_file)

    print("Alphabet in the file (letters only):")
    for char in unique_chars:
        print(f"{char} (U+{ord(char):04X})")

    print(f"\nNumber of unique words: {len(unique_words)}")

if __name__ == "__main__":
    main()
