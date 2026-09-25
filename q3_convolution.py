"""
CS5720 - Home Assignment 2
Question 3: Convolution with Different Strides and Padding

TensorFlow tf.nn.conv2d computes cross-correlation, which is also the operation
used by Keras Conv2D during CNN training. For the symmetric Laplacian kernel
given in the assignment, flipping the kernel would produce the same kernel.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

input_matrix = np.array([
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

x = tf.constant(input_matrix.reshape(1, 5, 5, 1))
k = tf.constant(kernel.reshape(3, 3, 1, 1))

def run_convolution(stride, padding):
    y = tf.nn.conv2d(
        x,
        filters=k,
        strides=[1, stride, stride, 1],
        padding=padding
    )
    return y.numpy()[0, :, :, 0]

print("Input Matrix:\n", input_matrix)
print("\nKernel:\n", kernel)

cases = [
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME")
]

for stride, padding in cases:
    output = run_convolution(stride, padding)
    print(f"\nStride = {stride}, Padding = '{padding}'")
    print(output)
    print("Output shape:", output.shape)
