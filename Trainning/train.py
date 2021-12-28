"""
@author: Minh Duc
@since: 12/21/2021 5:13 PM
@description:
@update:
"""

import json
import logging
import os
import pandas as pd
import nni

from Trainning.recommenders.utils.constants import *

import Trainning.recommenders.evaluation.python_evaluation as evaluation
from ncf_singlenode import NCF
from dataset import Dataset as NCFDataset
from Trainning.recommenders.utils.constants import SEED as DEFAULT_SEED

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ncf")


def loadModel(dataset: NCFDataset, model_type, n_epochs=10, learning_rate=5e-3, n_factors=8, checkPoint=None):
    model = NCF(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        model_type=model_type,

        n_epochs=n_epochs,
        learning_rate=learning_rate,
        n_factors=n_factors,

        seed=DEFAULT_SEED,
    )

    model.setData(dataset)
    if checkPoint:
        if model_type == "neumf":
            model.load(neumf_dir=checkPoint)
        elif model_type == "gmf":
            model.load(gmf_dir=checkPoint)
        elif model_type == "mlp":
            model.load(mlp_dir=checkPoint)
        else:
            raise "ModelType Invalid"
    return model


def ncf_training(model: NCF, dataset: NCFDataset, checkPoint=None):
    """
    Training NCF Model
    """
    logger.info("Start training...")
    model.fit(dataset)
    if checkPoint:
        model.save(checkPoint)
    logger.info("Finished Training")
    return model


def calculate_metrics(model, test_data, metrics_filename):
    metrics_dict = {}

    rating_metrics = evaluation.metrics
    predictions = [
        [row.userID, row.itemID, model.predict(row.userID, row.itemID)]
        for (_, row) in test_data.iterrows()
    ]
    predictions = pd.DataFrame(
        predictions, columns=["userID", "itemID", "prediction"]
    )
    predictions = predictions.astype(
        {"userID": "int64", "itemID": "int64", "prediction": "float64"}
    )
    print(predictions)
    for metric in rating_metrics:
        result = getattr(evaluation, metric)(test_data, predictions)
        metrics_dict[metric] = result
    # print(metrics_dict)
    nni.report_final_result(metrics_dict)

    # Save the metrics in a JSON file
    if metrics_filename:
        with open(metrics_filename, "w") as fp:
            temp_dict = metrics_dict.copy()
            json.dump(temp_dict, fp)


if __name__ == "__main__":
    model_type = ['gmf', 'mlp', 'neumf']
    for modelType in model_type:
        check_point = 'model_checkpoint_' + modelType
        train_data = pd.read_csv('data/ml-100k/u.data', delimiter='\t', names=DEFAULT_HEADER)
        test_data = pd.read_csv('data/ml-100k/u1.test', delimiter='\t', names=DEFAULT_HEADER)
        validation_data = pd.read_csv('data/ml-100k/u2.test', delimiter='\t', names=DEFAULT_HEADER)
        data = NCFDataset(train=train_data, validate=validation_data, seed=DEFAULT_SEED)

        # Create Model and Load the Parameters Checkpoint if exists
        model = loadModel(dataset=data, model_type=modelType, n_epochs=100, learning_rate=5e-3, n_factors=8,
                          checkPoint=check_point)

        # Training model
        # Comment this line if You want evaluate model without training
        ncf_training(model, dataset=data, checkPoint=check_point)

        # Model Evaluation with metrics
        calculate_metrics(model=model, test_data=test_data, metrics_filename='metrics_' + modelType + '.json')

