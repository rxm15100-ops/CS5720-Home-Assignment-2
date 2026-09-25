"""
CS5720 - Home Assignment 2
Question 1: LSTM Character-Level Text Generation

Dataset: Shakespeare text downloaded by TensorFlow/Keras.
The model learns to predict the next character from a fixed-length sequence.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load a public-domain Shakespeare text file.
path = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
text = open(path, "rb").read().decode("utf-8")
# Keep the run manageable for a student computer.
text = text[:500000]

# 2. Convert characters to integer sequences.
vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
encoded = np.array([char2idx[c] for c in text], dtype=np.int32)

SEQ_LEN = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Each training example has 100 input chars and 100 next-char targets.
char_dataset = tf.data.Dataset.from_tensor_slices(encoded)
sequences = char_dataset.batch(SEQ_LEN + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE, seed=42).batch(
    BATCH_SIZE, drop_remainder=True
).prefetch(tf.data.AUTOTUNE)

# 3. Define an LSTM-based RNN.
model = Sequential([
    Input(shape=(None,)),
    Embedding(len(vocab), 128),
    LSTM(256, return_sequences=True),
    Dense(len(vocab))
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.summary()

# 4. Train the model.
history = model.fit(
    dataset,
    epochs=10,
    callbacks=[EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)],
    verbose=1
)

def generate_text(model, start_string, num_generate=500, temperature=0.8):
    """Generate text one character at a time using temperature-scaled sampling."""
    input_ids = tf.expand_dims([char2idx[c] for c in start_string], 0)
    generated = []

    # Reset recurrent state by simply starting a fresh inference call sequence.
    for _ in range(num_generate):
        predictions = model(input_ids, training=False)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        generated.append(idx2char[predicted_id])

        # Keep a rolling context window.
        next_context = np.append(input_ids.numpy()[0], predicted_id)[-SEQ_LEN:]
        input_ids = tf.expand_dims(next_context, 0)

    return start_string + "".join(generated)

print("\n--- Generated text (temperature = 0.5) ---")
print(generate_text(model, "ROMEO: ", 400, 0.5))

print("\n--- Generated text (temperature = 1.0) ---")
print(generate_text(model, "ROMEO: ", 400, 1.0))

print("\nTemperature explanation:")
print("Lower temperature (for example 0.5) makes high-probability characters more likely,")
print("so output is more predictable. Higher temperature (for example 1.0 or above)")
print("flattens the effective distribution and increases variety/randomness, but may")
print("also increase spelling or grammatical errors.")
