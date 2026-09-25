# Home Assignment 2 - Video Presentation Script

## Opening
Hello, my name is Ramya Modala. This is my Home Assignment 2 for CS5720 Neural Network and Deep Learning at the University of Central Missouri. I will briefly demonstrate all five questions.

## Question 1 - RNN Text Generation
For Question 1, I implemented character-level text generation using an LSTM. I loaded Shakespeare text, created a character vocabulary, converted the text to integer sequences, and trained an embedding-LSTM-Dense model to predict the next character. The generation function samples one character at a time. I also compare temperature values. Lower temperature makes the output more predictable, while higher temperature increases randomness and diversity.

Run:
`python q1_rnn_text_generation.py`

Show the model summary, training output, and generated text.

## Question 2 - Sentiment Classification
For Question 2, I used the Keras IMDB dataset. I padded the reviews to a fixed length and trained an LSTM classifier with a sigmoid output for binary sentiment. After prediction, I generated a confusion matrix and classification report containing precision, recall, F1-score, and accuracy. The precision-recall balance changes when the decision threshold changes.

Run:
`python q2_sentiment_lstm.py`

Show the final test accuracy, classification report, and confusion matrix figure.

## Question 3 - Convolution
For Question 3, I used the exact 5 by 5 matrix and 3 by 3 kernel provided in the assignment. I used TensorFlow convolution with stride 1 and 2 and both VALID and SAME padding. VALID does not add zero padding, while SAME adds padding to preserve output size relative to the stride.

Run:
`python q3_convolution.py`

Show all four output feature maps.

## Question 4 - Edge Detection and Pooling
For Question 4 Task 1, I applied the assignment's Sobel-X and Sobel-Y filters to a grayscale sample image. The output shows the original image and detected directional edges. For Task 2, I created a random 4 by 4 matrix and applied 2 by 2 max pooling and average pooling.

Run:
`python q4_filters_pooling.py`

Show the Sobel figure and printed pooling matrices.

## Question 5 - AlexNet and ResNet
For Question 5, I implemented the simplified AlexNet architecture with the requested convolution, pooling, dense, dropout, and softmax layers. I also implemented a residual block with two 3 by 3 convolution layers and a skip connection, then used two blocks in a simple ResNet-like model.

Run:
`python q5_alexnet_resnet.py`

Show both model summaries.

## Closing
This completes all five questions for Home Assignment 2. The repository also contains the README, requirements file, source code, and generated outputs. Thank you.
