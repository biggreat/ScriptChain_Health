# Learnable Positional Encoding in Transformer Model

This Python script demonstrates the implementation of a learnable positional encoding method in a Transformer model using TensorFlow. The learnable positional encoding allows the model to learn and adapt the positional information during training, potentially improving its performance on tasks that require capturing fine-grained positional relationships.

## Overview

The code in this repository consists of the following components:

- `learnable_positional_encoding.py`: This script contains the implementation of the `LearnablePositionalEmbedding` class, which defines the learnable positional encoding layer. The positional encodings are represented as a trainable weight matrix, enabling the model to learn the positional information during training. The `EncoderLayer` class represents a single layer in the Transformer model, and the `Transformer` class defines the overall model architecture. It combines the learnable positional encoding layer, multi-head self-attention mechanism, and feed-forward network to create a complete Transformer model. This script generates a dummy dataset, compiles the model with appropriate loss and metrics, and trains the model on the dataset. Additionally, it includes a custom callback `CustomCallback` to monitor the changes in the learnable positional embeddings after each epoch.

## Getting Started

To get started with the code, follow these steps:

1. Install the required dependencies. The code requires TensorFlow (version 2.x) and NumPy. You can install them using pip.

2. Once the dependencies are installed, you can run the `learnable_positional_encoding.py` script. It demonstrates how to use the learnable positional encoding in training a Transformer model.

## Results and Observations

During training, you will notice that the positional encodings change after each epoch, reflecting the learning and adaptation of the model. 


