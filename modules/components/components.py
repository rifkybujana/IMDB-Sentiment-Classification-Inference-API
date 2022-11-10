"""
Modules contains TFX components for the pipeline

Contains the following methods:
    init_components,
    data_ingestation,
    data_validation,
    data_transform,
    tuner,
    trainer,
    evaluator,
    pusher

Public variable:
    components: TFX Components (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
"""

from collections import namedtuple

import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
    Tuner
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)
from modules.utility import path, data_features

DataValidationComponents = namedtuple("DataValidationComponents", [
    "statistics_gen",
    "schema_gen",
    "example_validator"
])
EvaluatorComponents = namedtuple("EvaluatorComponents", [
    "resolver",
    "evaluator"
])

def _data_ingestation():
    """
    Generate data ingestation component for the pipeline
    """
    output = example_gen_pb2.Output(
        split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    return CsvExampleGen(
        input_base=path.DATA_ROOT,
        output_config=output
    )

def _data_validation(example_gen):
    """
    Generate data validation component for the pipeline
    """
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    return DataValidationComponents(
        statistics_gen,
        schema_gen,
        example_validator
    )

def _data_transform(example_gen, schema_gen):
    """
    Generate data transformation component for the pipeline
    """
    return Transform(
        examples=example_gen.outputs['examples'],
        schema= schema_gen.outputs['schema'],
        module_file=path.TRANSFORM_MODULE_FILE
    )

def _tuner(
    transform,
    schema_gen,
    training_steps,
    eval_steps
):
    """
    Generate model tuner to tune hyperparameter of the model
    """
    return Tuner(
        module_file=path.TUNER_MODULE_FILE,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

def _trainer(
    transform,
    schema_gen,
    training_steps,
    eval_steps,
    tuner
):
    """
    Generate model trainer for the pipeline
    """
    return Trainer(
        module_file=path.TRAINER_MODULE_FILE,
        examples = transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        # This will be passed to `run_fn`.
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=eval_steps)
    )

def _evaluator(example_gen, trainer):
    """
    Generate model resolver and evaluator for the pipeline
    """
    model_resolver = Resolver(
        strategy_class= LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=data_features.FEATURE_KEY)
    ]

    metrics_specs = [
        tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value':0.5}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value':0.0001})
                        )
                )
            ])
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=data_features.LABEL_KEY)],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

    return EvaluatorComponents(
        model_resolver,
        evaluator
    )

def _pusher(trainer, evaluator):
    """
    Generate pusher components for the pipeline
    """
    return Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=path.SERVING_MODEL_DIR
            )
        ),
    )

def init(training_steps, eval_steps):
    """
    Generate pipeline components
    """
    example_gen = _data_ingestation()
    data_validation = _data_validation(example_gen)
    transform = _data_transform(example_gen, data_validation.schema_gen)
    tuner = _tuner(
        transform,
        data_validation.schema_gen,
        training_steps,
        eval_steps
    )
    trainer = _trainer(
        transform,
        data_validation.schema_gen,
        training_steps,
        eval_steps,
        tuner
    )
    evaluator = _evaluator(example_gen, trainer)
    pusher = _pusher(trainer, evaluator.evaluator)

    return (
        example_gen,
        data_validation.statistics_gen,
        data_validation.schema_gen,
        data_validation.example_validator,
        transform,
        tuner,
        trainer,
        evaluator.resolver,
        evaluator.evaluator,
        pusher
    )
