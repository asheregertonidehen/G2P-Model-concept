import sys

def clean_text(text):
    # Characters to remove
    chars_to_remove = ['ô', 'ö', 'ü', 'Ş', 'š']
    for ch in chars_to_remove:
        text = text.replace(ch, '')
    return text

def clean_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            cleaned_line = clean_text(line)
            fout.write(cleaned_line)

def main():
    if len(sys.argv) != 3:
        print("Usage: python text-cleaner.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    clean_file(input_file, output_file)
    print(f"Cleaned file saved as {output_file}")

if __name__ == "__main__":
    main()
