import sys
import unicodedata
import re

def extract_alphabet_and_words(file_path):
    unique_chars = set()
    unique_words = set()

    # Regex to extract only words made of letters (no digits/punctuation)
    word_pattern = re.compile(r'\b[^\W\d_]+\b', re.UNICODE)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Collect unique letters
            for char in line:
                if unicodedata.category(char).startswith('L'):
                    unique_chars.add(char)
            
            # Extract unique words (lowercased)
            words = word_pattern.findall(line.lower())
            unique_words.update(words)

    return sorted(unique_chars), sorted(unique_words)

def main():
    if len(sys.argv) != 3:
        print("Usage: python edo_counter.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    unique_chars, unique_words = extract_alphabet_and_words(input_file)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("=== Unique Letters (Edo Alphabet) ===\n")
        for char in unique_chars:
            f_out.write(char + '\n')
        
        f_out.write("\n=== Unique Words ===\n")
        for word in unique_words:
            f_out.write(word + '\n')

    print(f"Extracted {len(unique_chars)} unique letters and {len(unique_words)} unique words into '{output_file}'.")

if __name__ == "__main__":
    main()
