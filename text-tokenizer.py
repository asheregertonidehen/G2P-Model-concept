import sys
import re
from collections import Counter

def tokenize_words(text):
    #Tokenize text into words (preserving diacritics, removing punctuation)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation, keep diacritics
    return text.split()

def tokenize_characters(text):
    #Tokenize text into characters (graphemes)
    tokens = []
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    for word in text.split():
        tokens.extend(list(word))  # Break each word into characters
    return tokens

def count_frequencies(tokens):
    #Count token frequencies.
    return Counter(tokens)

def main():
    if len(sys.argv) != 3:
        print("Usage: python edo_tokenizer.py <input_file> <mode>")
        print("Modes: word or char")
        sys.exit(1)

    input_file = sys.argv[1]
    mode = sys.argv[2]

    # Read file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize based on mode
    if mode == "word":
        tokens = tokenize_words(text)
    elif mode == "char":
        tokens = tokenize_characters(text)
    else:
        print("Invalid mode. Use 'word' or 'char'.")
        sys.exit(1)

    frequencies = count_frequencies(tokens)

    for token, count in frequencies.most_common(1000):  # Top 50
        print(f"{token}: {count}")

if __name__ == "__main__":
    main()
