"""
CS5720 - Home Assignment 2
Question 4: CNN Feature Extraction with Filters and Pooling

Task 1: Sobel edge detection using NumPy + OpenCV
Task 2: Max pooling and average pooling using TensorFlow/Keras
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D

# ---------------- Task 1: Sobel Edge Detection ----------------
# Use a generated grayscale sample image so the script is self-contained.
# It contains a rectangle, circle, and diagonal line with clear edges.
image = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(image, (35, 35), (130, 130), 180, -1)
cv2.circle(image, (185, 75), 40, 255, -1)
cv2.line(image, (30, 220), (225, 150), 220, 8)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

edge_x = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, sobel_x)
edge_y = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, sobel_y)

# Absolute values make positive and negative edge responses visible.
edge_x_display = cv2.convertScaleAbs(edge_x)
edge_y_display = cv2.convertScaleAbs(edge_y)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edge_x_display, cmap="gray")
plt.title("Sobel-X")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge_y_display, cmap="gray")
plt.title("Sobel-Y")
plt.axis("off")

plt.tight_layout()
plt.savefig("q4_sobel_results.png", dpi=150)
plt.show()

# ---------------- Task 2: Pooling ----------------
np.random.seed(42)
input_matrix = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)
x = tf.constant(input_matrix.reshape(1, 4, 4, 1))

max_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

print("\nOriginal 4x4 Matrix:")
print(input_matrix)

print("\n2x2 Max-Pooled Matrix:")
print(max_pool.numpy()[0, :, :, 0])

print("\n2x2 Average-Pooled Matrix:")
print(avg_pool.numpy()[0, :, :, 0])
