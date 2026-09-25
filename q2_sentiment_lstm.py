"""
CS5720 - Home Assignment 2
Question 2: Sentiment Classification Using an LSTM and the IMDB dataset
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(42)
tf.random.set_seed(42)

VOCAB_SIZE = 10000
MAX_LEN = 250

# 1. Load IMDB dataset. Reviews are already tokenized as integer word indices.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# 2. Pad all sequences to a common length.
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)

# 3. LSTM sentiment classifier.
model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(VOCAB_SIZE, 128, mask_zero=True),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# 4. Confusion matrix and classification report.
probabilities = model.predict(x_test, batch_size=256, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Negative", "Positive"],
    digits=4
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("IMDB Sentiment Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("q2_confusion_matrix.png", dpi=150)
plt.show()

# 5. Precision-recall tradeoff interpretation.
print("""
Precision-Recall Tradeoff:
Precision measures how many reviews predicted as a class are actually that class.
Recall measures how many real examples of that class the model successfully finds.
Changing the probability threshold changes this balance. A stricter threshold can
increase precision but may miss more true positives (lower recall), while a lower
threshold can increase recall but may create more false positives. The preferred
balance depends on the application and the cost of each type of error.
""")
