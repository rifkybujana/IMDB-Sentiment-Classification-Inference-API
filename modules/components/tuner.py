"""
Model hyperparameter tuner
"""

import keras_tuner

import tensorflow_transform as tft

from tfx.components.tuner.component import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
from modules.components import model_builder, transform, trainer
from modules.utility import data_features

def tuner_fn(fn_args: FnArgs):
    """Build the tuner using the KerasTuner API.
    Args:
        fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
        A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                        model , e.g., the training and validation dataset. Required
                        args depend on the above tuner's implementation.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = trainer.input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = trainer.input_fn(fn_args.eval_files, tf_transform_output, 10)

    model_builder.vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transform.transformed_name(data_features.FEATURE_KEY)]
                for i in list(train_set)]])
    tuner = keras_tuner.RandomSearch(
        model_builder.model_builder,
        max_trials=6,
        hyperparameters=model_builder.get_hyperparameters(),
        allow_new_entries=False,
        objective=keras_tuner.Objective('val_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='imdb_sentiment_classification')

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
