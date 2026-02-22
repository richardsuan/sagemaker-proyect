import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter
)

session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(

    entry_point="train_model.py",

    role=role,

    instance_count=1,
    instance_type="ml.m5.large",

    framework_version="1.13",
    py_version="py39"
)

hyperparameter_ranges = {

    "lr": ContinuousParameter(1e-5, 1e-2),
    "batch_size": IntegerParameter(16,64),
    "epochs": IntegerParameter(3,10)
}

objective_metric_name = "validation:accuracy"

tuner = HyperparameterTuner(

    estimator=estimator,

    objective_metric_name=objective_metric_name,
    hyperparameter_ranges=hyperparameter_ranges,

    max_jobs=8,
    max_parallel_jobs=2
)

tuner.fit({

    "train": "s3://sagemaker-us-east-1-220044031696/dog-project"
})
