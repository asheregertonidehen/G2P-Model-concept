import sys
import re

# Edo orthography → IPA dictionary (based on your image)
# Order matters: digraphs first, then single letters
G2P_MAPPING = {
    'gb': 'ɡb',
    'gh': 'ɣ',
    'kh': 'x',
    'kp': 'kp',
    'mw': 'ʋ',   # Usually represents labialized /m/ or /ʋ/
    'rh': 'ɾ',
    'rr': 'ɽ',
    'vb': 'ʋ',
    'ẹ': 'ɛ',
    'ọ': 'ɔ',
    'a': 'a',
    'b': 'b',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'y': 'j',
    'z': 'z'
}

# Lowercase everything and sort digraphs first
DIGRAPHS = sorted([k for k in G2P_MAPPING.keys() if len(k) > 1], key=len, reverse=True)
SINGLE_CHARS = [k for k in G2P_MAPPING.keys() if len(k) == 1]

# Combines all keys (digraphs + letters) in order of length
ALL_KEYS = DIGRAPHS + SINGLE_CHARS
G2P_PATTERN = re.compile('|'.join(re.escape(k) for k in ALL_KEYS))

# Function to convert Edo word to IPA transcription
def convert_word_to_ipa(word):
    word = word.lower()
    return G2P_PATTERN.sub(lambda m: G2P_MAPPING.get(m.group(0), ''), word)

# File processing function
def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            word = line.strip()
            if not word:
                continue
            ipa = convert_word_to_ipa(word)
            fout.write(f"{word}|{ipa}\n")

# CLI interface
def main():
    if len(sys.argv) != 3:
        print("Usage: python text-IPAconvert.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_file(input_file, output_file)
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()
