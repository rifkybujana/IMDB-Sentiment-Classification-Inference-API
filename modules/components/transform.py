"""
Modules used to transform data into the format that model accept
and do some preprocessing process to the data.
"""

import tensorflow as tf

from modules.utility import data_features

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """preprocess the data"""
    outputs = {}

    # here we dont do anything to the data as the data
    # will be preprocessed in the model embeddings
    outputs[
        transformed_name(data_features.FEATURE_KEY)
    ] = inputs[data_features.FEATURE_KEY]

    # cast the label to make sure it is a int64 data type
    outputs[
        transformed_name(data_features.LABEL_KEY)
    ] = tf.cast(inputs[data_features.LABEL_KEY], tf.int64)

    return outputs
