
import unicodedata
import sys


readFile = sys.argv[1]
writeFile = sys.argv[2]

if len(sys.argv) < 3:
    print("Usage: python text-normalizer.py <input_file> <output_file>")
    sys.exit(1)

def normalize(text):
    return unicodedata.normalize('NFC', text) 

with open(readFile, 'r', encoding='utf-8') as file:
    lines = file.readlines()

normalized_lines = [normalize(line.strip()) for line in lines]

with open(writeFile, 'w', encoding='utf-8') as f:
    for line in normalized_lines:
        f.write(line + '\n')
print(f"File created successfully: {writeFile}")