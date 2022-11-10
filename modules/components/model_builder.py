"""
Model architecture builder
"""

import keras_tuner

import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization

from modules.utility import data_features
from modules.components import transform

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

def get_hyperparameters() -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hparams = keras_tuner.HyperParameters()
    # Defines search space.
    hparams.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
    hparams.Int('num_layers', 1, 3, default=2)
    return hparams

def model_builder(hparams: keras_tuner.HyperParameters) -> tf.keras.Model:
    """
    Creates a DNN Keras model.
    Args:
        hparams: Holds HyperParameters for tuning.
    Returns:
        A Keras Model.
    """

    embedding_dim=16

    inputs = tf.keras.Input(
        shape=(1,),
        name=transform.transformed_name(
            data_features.FEATURE_KEY
        ),
        dtype=tf.string
    )
    reshaped_narrative = tf.reshape(inputs, [-1])

    net = vectorize_layer(reshaped_narrative)
    net = Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(net)
    net = GlobalAveragePooling1D()(net)

    for _ in range(int(hparams.get('num_layers'))):
        net = Dense(32, activation='relu')(net)
    net = Dense(1, activation="sigmoid")(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams.get('learning_rate')
        ),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        ),
        metrics=['accuracy']
    )
    model.summary()
    return model
