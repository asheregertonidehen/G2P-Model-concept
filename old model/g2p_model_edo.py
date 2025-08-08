# g2p_model_edo.py - FIXED VERSION

import tensorflow as tf
import numpy as np
import string
import os
import sys

# --- 1. CONFIGURATION AND DATA LOADING ---
# Path to your metadata file. Replace 'new_metadata.txt' with your actual file name.
METADATA_FILE = 'new_metadata.txt'
MODEL_FILE = 'edo_g2p_model.keras'

# Mock data for demonstration purposes.
if not os.path.exists(METADATA_FILE):
    print("Creating a mock new_metadata.txt file for demonstration.")
    mock_data = [
        "igun|i g u n",
        "omoi|o m o i",
        "ebe|ɛ b ɛ",
        "osa|o s a",
        "ukpogho|u k p o ɣ o",
        "uwa|u w a",
        "rhie|r i ɛ",
        "obo|ɔ b ɔ",
        "vbe|ʋ b ɛ",
        "gha|ɣ a",
        "Edo|e d o",
        "okaro|o k a r o",
        "eshi|e ʃ i",
        "evbo|e ʋ b o"
    ]
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(mock_data))
else:
    print(f"Using existing metadata file: {METADATA_FILE}")


def load_data(file_path):
    """
    Loads grapheme-to-phoneme pairs from a file.
    The file is expected to have 'grapheme|phoneme_sequence' on each line.
    """
    graphemes = []
    phonemes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                grapheme_seq, phoneme_seq = line.split('|')
                # Split the phoneme sequence into individual phonemes
                graphemes.append(list(grapheme_seq.strip()))
                phonemes.append(phoneme_seq.strip().split(' '))
    return graphemes, phonemes


# --- 2. VOCABULARY AND PREPROCESSING ---
def create_vocabularies(grapheme_sequences, phoneme_sequences):
    """Creates mappings for graphemes and phonemes."""
    all_graphemes = set(char for seq in grapheme_sequences for char in seq)
    all_phonemes = set(char for seq in phoneme_sequences for char in seq)

    grapheme_tokens = sorted(list(all_graphemes)) + ['<pad>']
    phoneme_tokens = sorted(list(all_phonemes)) + ['<sos>', '<eos>', '<pad>']

    grapheme_to_id = {char: i for i, char in enumerate(grapheme_tokens)}
    id_to_grapheme = {i: char for i, char in enumerate(grapheme_tokens)}

    phoneme_to_id = {char: i for i, char in enumerate(phoneme_tokens)}
    id_to_phoneme = {i: char for i, char in enumerate(phoneme_tokens)}

    max_grapheme_len = max(len(seq) for seq in grapheme_sequences)
    max_phoneme_len = max(len(seq) for seq in phoneme_sequences) + 2

    return (grapheme_to_id, id_to_grapheme, phoneme_to_id, id_to_phoneme,
            max_grapheme_len, max_phoneme_len)


def preprocess_data(grapheme_seqs, phoneme_seqs, grapheme_to_id, phoneme_to_id,
                    max_grapheme_len, max_phoneme_len):
    """Converts sequences to padded integer arrays for model input."""
    encoder_input_data = np.zeros(
        (len(grapheme_seqs), max_grapheme_len), dtype='int32')
    decoder_input_data = np.zeros(
        (len(phoneme_seqs), max_phoneme_len), dtype='int32')
    decoder_target_data = np.zeros(
        (len(phoneme_seqs), max_phoneme_len, len(phoneme_to_id)), dtype='int32')

    for i, (g_seq, p_seq) in enumerate(zip(grapheme_seqs, phoneme_seqs)):
        for t, grapheme in enumerate(g_seq):
            encoder_input_data[i, t] = grapheme_to_id[grapheme]

        decoder_input_data[i, 0] = phoneme_to_id['<sos>']
        for t, phoneme in enumerate(p_seq):
            decoder_input_data[i, t + 1] = phoneme_to_id[phoneme]
            decoder_target_data[i, t, phoneme_to_id[phoneme]] = 1
        decoder_target_data[i, len(p_seq), phoneme_to_id['<eos>']] = 1

    return encoder_input_data, decoder_input_data, decoder_target_data


# --- 3. BUILD AND TRAIN THE MODEL ---
def build_and_train_model():
    """Builds, trains, and saves the G2P model."""
    grapheme_sequences, phoneme_sequences = load_data(METADATA_FILE)
    print(f"Loaded {len(grapheme_sequences)} data pairs.")

    (grapheme_to_id, id_to_grapheme, phoneme_to_id, id_to_phoneme,
     max_grapheme_len, max_phoneme_len) = create_vocabularies(
        grapheme_sequences, phoneme_sequences)

    encoder_input_data, decoder_input_data, decoder_target_data = preprocess_data(
        grapheme_sequences, phoneme_sequences, grapheme_to_id, phoneme_to_id,
        max_grapheme_len, max_phoneme_len)

    # Hyperparameters
    LATENT_DIM = 256
    EMBEDDING_DIM = 128
    EPOCHS = 50
    BATCH_SIZE = 32

    # Encoder
    encoder_inputs = tf.keras.layers.Input(shape=(max_grapheme_len,))
    encoder_embedding = tf.keras.layers.Embedding(
        len(grapheme_to_id), EMBEDDING_DIM)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(LATENT_DIM, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(
        len(phoneme_to_id), EMBEDDING_DIM)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(
        len(phoneme_to_id), activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # The full model.
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print("\nStarting model training...")
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2
    )

    # Save the entire model for later use.
    model.save(MODEL_FILE)
    print(f"\nModel saved to '{MODEL_FILE}'")


# --- 4. INFERENCE MODEL FOR PREDICTION ---
def predict_ipa(input_word, inference_encoder_model, inference_decoder_model,
                grapheme_to_id, id_to_phoneme, phoneme_to_id, max_grapheme_len, max_phoneme_len):
    """
    Predicts the IPA representation for a single word.
    """
    input_seq = np.zeros((1, max_grapheme_len), dtype='int32')
    for t, char in enumerate(input_word.lower()):
        input_seq[0, t] = grapheme_to_id.get(char, grapheme_to_id['<pad>'])

    states_value = inference_encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1), dtype='int32')
    target_seq[0, 0] = phoneme_to_id['<sos>']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = inference_decoder_model.predict(
            [target_seq] + states_value
        )
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id_to_phoneme[sampled_token_index]
        decoded_sentence.append(sampled_char)

        if sampled_char == '<eos>' or len(decoded_sentence) > max_phoneme_len:
            stop_condition = True

        target_seq = np.zeros((1, 1), dtype='int32')
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(decoded_sentence).replace(' <eos>', '').replace(' <sos>', '').strip()


def run_interactive_prediction():
    """Loads the model and runs an interactive loop for prediction."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found. Please train the model first.")
        sys.exit(1)

    print("\nLoading the saved model...")
    model = tf.keras.models.load_model(MODEL_FILE)

    grapheme_sequences, phoneme_sequences = load_data(METADATA_FILE)
    (grapheme_to_id, id_to_grapheme, phoneme_to_id, id_to_phoneme,
     max_grapheme_len, max_phoneme_len) = create_vocabularies(
        grapheme_sequences, phoneme_sequences)

    # --- FIXED INFERENCE MODEL RECONSTRUCTION ---
    # Use the correct layer names as defined during training
    encoder_inputs = model.input[0]
    # Use the actual layer names: 'encoder_lstm' and 'decoder_lstm'
    encoder_lstm = model.get_layer('encoder_lstm')
    _, state_h_enc, state_c_enc = encoder_lstm(model.get_layer('embedding')(encoder_inputs)) 
    encoder_states = [state_h_enc, state_c_enc]
    inference_encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_dense = model.get_layer('decoder_dense')

    decoder_state_input_h = tf.keras.layers.Input(shape=(256,), name='input_h')
    decoder_state_input_c = tf.keras.layers.Input(shape=(256,), name='input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embedding = model.get_layer('embedding_1')(decoder_inputs)
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    inference_decoder_model = tf.keras.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    print("Model loaded successfully. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("Enter an Edo word: ")
            if user_input.lower() == 'quit':
                break
            if not user_input.strip():
                continue

            predicted_ipa = predict_ipa(
                user_input,
                inference_encoder_model,
                inference_decoder_model,
                grapheme_to_id, id_to_phoneme, phoneme_to_id,
                max_grapheme_len, max_phoneme_len
            )
            print(f"IPA: {predicted_ipa}\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'predict':
        run_interactive_prediction()
    else:
        build_and_train_model()
        print("\nTraining complete. To make predictions, run 'python g2p_model_edo.py predict'")
