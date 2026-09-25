"""
CS5720 - Home Assignment 2
Question 5: Implementing and Comparing CNN Architectures
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Add, Activation
)

# ---------------- Task 1: Simplified AlexNet ----------------
# 'same' padding on later convolutions keeps spatial dimensions valid
# while preserving the layer/filter/stride requirements in the assignment.
alexnet = Sequential([
    Input(shape=(227, 227, 3)),
    Conv2D(96, (11, 11), strides=4, activation="relu", padding="valid"),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(256, (5, 5), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(384, (3, 3), activation="relu", padding="same"),
    Conv2D(384, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Flatten(),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
], name="Simplified_AlexNet")

print("\n================ ALEXNET SUMMARY ================\n")
alexnet.summary()

# ---------------- Task 2: Residual Block + ResNet ----------------
def residual_block(input_tensor, filters=64):
    """
    Two 3x3 convolutions plus an identity skip connection.
    The first convolution uses ReLU. The second is linear before Add,
    then ReLU is applied after the skip connection as requested.
    """
    x = Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
    x = Conv2D(filters, (3, 3), padding="same", activation=None)(x)

    # Input has the same number of channels (64), so identity addition is valid.
    x = Add()([input_tensor, x])
    x = Activation("relu")(x)
    return x

inputs = Input(shape=(64, 64, 3))
x = Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(inputs)
x = residual_block(x, 64)
x = residual_block(x, 64)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

resnet_model = Model(inputs, outputs, name="Simple_ResNet")

print("\n================ RESNET-LIKE SUMMARY ================\n")
resnet_model.summary()
