"""
Trainer modules to train the model
"""

import os
import absl
import keras_tuner

import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from modules.components import model_builder, transform
from modules.utility import data_features

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transform.transformed_name(data_features.LABEL_KEY))
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(data_features.LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        # get predictions using the transformed features
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    """Train the model based on given args.
    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    log_dir = os.path.join(
        os.path.dirname(fn_args.serving_model_dir),
        'logs'
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        patience=10
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True
    )
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    model_builder.vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transform.transformed_name(data_features.FEATURE_KEY)]
                for i in list(train_set)]])

    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # This is a shown case when hyperparameters is decided and Tuner is removed
        # from the pipeline. User can also inline the hyperparameters directly in
        # _build_keras_model.
        hparams = model_builder.get_hyperparameters()
    absl.logging.info('HyperParameters for training: %s' % hparams.get_config())
    # Build the model
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model = model_builder.model_builder(hparams)
    model = model_builder.model_builder(hparams)
    # Train the model
    model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [
                tensorboard_callback,
                early_stopping,
                model_checkpoint
            ],
            steps_per_epoch = 10,
            validation_steps= 10,
            epochs=100)

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
