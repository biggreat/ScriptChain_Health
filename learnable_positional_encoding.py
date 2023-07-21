"""
Design a learnable positional encoding method
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class LearnablePositionalEmbedding(Layer):
    """
    Custom layer for learnable positional embedding.
    """

    def __init__(self, sequence_length, output_dim, **kwargs):
        """
        Initialize the object.

        :param sequence_length: The length of the sequence.
        :param output_dim: The dimension of the output.
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def build(self, input_shape):
        """
        Build the layer. This method is called once before the first call to 'call'.

        :param input_shape: The shape of the input tensor.
        """
        self.pos_encoding = self.add_weight(
            shape=(self.sequence_length, self.output_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        """
        Call the layer on the inputs. This is where the actual logic of the layer lives.

        :param inputs: The input tensor.
        :return: The output tensor (inputs + positional encoding).
        """
        return inputs + self.pos_encoding


class EncoderLayer(Layer):
    """
    Encoder layer class.
    """

    def __init__(self, num_heads, embed_dim, ff_dim):
        """
        Initialize the object.

        :param num_heads: The number of attention heads.
        :param embed_dim: The dimension of the embedding.
        :param ff_dim: The dimension of the feed-forward network inside EncoderLayer.
        """
        super(EncoderLayer, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Add dropout
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs):
        """
        Call the model on new inputs. This is where the actual logic of applying the model to an input data.

        :param inputs: The input tensor.
        :return: The output tensor (model output).
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class Transformer(tf.keras.Model):
    """
    Custom model class.
    """

    def __init__(self, num_classes, sequence_length, num_heads, embed_dim, ff_dim):
        """
        Initialize the object.

        :param num_classes: The number of classes (also the dimension of the output).
        :param sequence_length: The length of the sequence.
        :param num_heads: The number of attention heads.
        :param embed_dim: The dimension of the embedding.
        :param ff_dim: The dimension of the feed-forward network inside EncoderLayer.
        """
        super(Transformer, self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim=num_classes, output_dim=embed_dim)
        self.pos_enc = LearnablePositionalEmbedding(sequence_length=sequence_length, output_dim=embed_dim)
        self.enc_layer = EncoderLayer(num_heads=num_heads, embed_dim=embed_dim, ff_dim=ff_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        """
        Call the model on new inputs. This is where the actual logic of applying the model to an input data.

        :param inputs: The input tensor.
        :param **kwargs:
        :return: The output tensor (model output).
        """
        x = self.embed(inputs)
        x = self.pos_enc(x)
        x = self.enc_layer(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to monitor changes in the learnable positional embedding after each epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each epoch.

        :param epoch: The current epoch number.
        :param logs: A dictionary of logs.
        """
        positional_embeddings = self.model.pos_enc.get_weights()[0]
        print("Positional embeddings after epoch {}:".format(epoch + 1))
        print(positional_embeddings)


# Define parameters
sequence_length = 100
num_classes = 30
batch_size = 32

# Create dummy input and target data
X = np.random.randint(num_classes, size=(1000, sequence_length))
y = np.random.randint(2, size=(1000,))

# Create a tf.data.Dataset from tensor slices
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

# Define the model
model = Transformer(num_classes=num_classes, sequence_length=sequence_length, num_heads=2, embed_dim=30, ff_dim=30)

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(dataset, epochs=10, callbacks=[CustomCallback()])
