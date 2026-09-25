# CS5720 Neural Network and Deep Learning
## Home Assignment 2 - Fall 2026

**Student Name:** Ramya Modala  
**Course:** CS5720 Neural Network and Deep Learning  
**University:** University of Central Missouri  
**Assignment:** Home Assignment 2

## Overview

This repository contains the complete implementation for Home Assignment 2. The work covers recurrent neural networks, LSTM text generation, sentiment classification, convolution operations, Sobel edge detection, pooling, AlexNet, and residual networks.

## Files

- `q1_rnn_text_generation.py` - Character-level LSTM text generation using Shakespeare text.
- `q2_sentiment_lstm.py` - IMDB positive/negative sentiment classification with an LSTM, confusion matrix, and classification report.
- `q3_convolution.py` - 5x5 input convolution using the assignment's 3x3 kernel with stride 1/2 and VALID/SAME padding.
- `q4_filters_pooling.py` - Sobel-X/Sobel-Y edge detection plus max/average pooling.
- `q5_alexnet_resnet.py` - Simplified AlexNet and ResNet-like model implementations.
- `requirements.txt` - Python dependencies.
- `VIDEO_SCRIPT.md` - Short presentation guide.
- `CS5720_Home_Assignment_2_Completed.docx` - Written explanation of the assignment.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Questions

```bash
python q1_rnn_text_generation.py
python q2_sentiment_lstm.py
python q3_convolution.py
python q4_filters_pooling.py
python q5_alexnet_resnet.py
```

Q1 and Q2 download public datasets the first time they run, so an internet connection is needed for the initial download.

## Question 1 - RNN Text Generation

The program downloads Shakespeare text and converts every character into an integer ID. An embedding layer converts the IDs into dense vectors. An LSTM learns sequential patterns and a Dense layer predicts logits for the next character.

Text is generated one character at a time. Temperature scaling divides the prediction logits before sampling. A lower temperature makes generation more conservative and predictable. A higher temperature produces more diverse/random output but can also increase errors.

## Question 2 - Sentiment Classification

The Keras IMDB dataset is loaded with a vocabulary limit of 10,000 words. Reviews are padded/truncated to 250 tokens. The classifier uses an Embedding layer, an LSTM, Dropout, and a sigmoid output neuron.

Predictions are converted to labels with a 0.5 threshold. The program prints the confusion matrix and classification report containing precision, recall, F1-score, and accuracy.

Precision and recall can trade off when the decision threshold changes. Higher precision means fewer false-positive predictions; higher recall means fewer positive examples are missed. Which one matters more depends on the application.

## Question 3 - Convolution

Input:
```
1  2  3  4  5
6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

Kernel:
```
 0  1  0
 1 -4  1
 0  1  0
```

The program evaluates:
- Stride 1, VALID
- Stride 1, SAME
- Stride 2, VALID
- Stride 2, SAME

VALID does not add zero-padding, so the feature map becomes smaller. SAME adds padding so output spatial dimensions are approximately the input dimensions divided by the stride.

## Question 4 - CNN Feature Extraction

Task 1 applies the exact Sobel-X and Sobel-Y kernels from the assignment. Sobel-X emphasizes changes in the horizontal image direction and therefore highlights vertical edges. Sobel-Y emphasizes changes in the vertical direction and therefore highlights horizontal edges.

Task 2 generates a reproducible random 4x4 matrix. Max pooling keeps the maximum value from each 2x2 region, while average pooling returns the mean of each region.

## Question 5 - AlexNet and ResNet

The AlexNet implementation contains the requested convolution, max-pooling, fully connected, dropout, and 10-class softmax layers.

The ResNet-like model begins with a 64-filter 7x7 convolution with stride 2. Two residual blocks are then applied. Each residual block contains two 3x3 convolutions and an identity skip connection. The skip connection allows information and gradients to pass more directly through the network, helping deeper networks train.

## Submission Checklist

- [ ] Run every Python file successfully.
- [ ] Save screenshots/output from each question.
- [ ] Confirm `q2_confusion_matrix.png` is generated.
- [ ] Confirm `q4_sobel_results.png` is generated.
- [ ] Push source code and README to your GitHub repository.
- [ ] Confirm your student information is in this README.
- [ ] Record the demonstration video using `VIDEO_SCRIPT.md`.
- [ ] Submit the GitHub repository link and video on Brightspace.
