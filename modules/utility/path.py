"""
Module to store every path used for the pipeline
    ABS_PATH: absolute path of this project folder
    PIPELINE_NAME: the name of the project pipeline
    SCHEMA_PIPELINE_NAME: the name of the project pipeline schema
    PIPELINE_ROOT: Path to save the pipelines
    METADATA_PATH: Path to a SQLite DB file to use as an MLMD storage.
    SERVING_MODEL_DIR: Output directory where created models from the pipeline will be exported
    TRANSORM_MODULE_FILE: File holder to preprocess the data
    TUNER_MODULE_FILE: File holder to tune the model
    TRAINER_MODULE_FILE: File holder to train the model
    DATA_ROOT: Data path
"""

import os

def get_root_path(path):
    """
    Get the absolute path of the project
    """
    path = os.path.split(path)
    if path[1] == "imdb_sentiment_classification":
        return os.path.join(path[0], path[1])

    return get_root_path(path[0])

ABS_PATH = get_root_path(os.getcwd())

PIPELINE_NAME = "rifkybujana-pipeline"
SCHEMA_PIPELINE_NAME = "rifkybujana-tfdv-schema"

PIPELINE_ROOT = os.path.join(ABS_PATH, 'pipelines', PIPELINE_NAME)

METADATA_PATH = os.path.join(ABS_PATH, 'metadata', PIPELINE_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join(ABS_PATH, 'serving_model', PIPELINE_NAME)

TRANSFORM_MODULE_FILE = os.path.join(ABS_PATH, "modules", "components", "transform.py")

TUNER_MODULE_FILE = os.path.join(ABS_PATH, "modules", "components", "tuner.py")

TRAINER_MODULE_FILE = os.path.join(ABS_PATH, "modules", "components", "trainer.py")

DATA_ROOT = os.path.join(ABS_PATH, "data")
