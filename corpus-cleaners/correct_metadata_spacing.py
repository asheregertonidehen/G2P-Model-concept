# correct_metadata_spacing.py

import os

# --- Configuration ---
INPUT_FILE = 'metadata.txt'
OUTPUT_FILE = 'new_metadata.txt'

# --- Script Logic ---

if not os.path.exists(INPUT_FILE):
    print(f"Error: The input file '{INPUT_FILE}' was not found.")
else:
    print(f"Reading from '{INPUT_FILE}'...")
    corrected_lines = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if '|' in line:
                try:
                    # Split the line into the word and the IPA sequence
                    word, ipa_sequence = line.split('|', 1)
                    
                    # Split the IPA sequence into individual characters and join with a space.
                    # This ensures single-character phonemes are correctly separated.
                    corrected_ipa = ' '.join(list(ipa_sequence.replace(' ', ''))).strip()
                    
                    # Create the new, corrected line
                    corrected_line = f"{word}|{corrected_ipa}"
                    corrected_lines.append(corrected_line)
                except ValueError:
                    print(f"Skipping malformed line: {line}")
            else:
                # Keep lines that don't contain a '|' (e.g., blank lines)
                corrected_lines.append(line)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(corrected_lines))

    print(f"Done! The corrected file has been saved as '{OUTPUT_FILE}'.")
    print("You can now rename this file to metadata.txt to use it with the G2P model.")

