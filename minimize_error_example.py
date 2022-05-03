"""Optimizing model accuracy."""

import pandas as pd
from sklearn.datasets import make_regression
from ax import optimize
from ax.utils.measurement.synthetic_functions import branin
from sklearn.ensemble import RandomForestRegressor

# X_train, y_train =

parameters = [
    {"name": "x1", "type": "range", "bounds": [-5.0, 10.0],},
    {"name": "x2", "type": "range", "bounds": [0.0, 10.0],},
]

param_names = [p["name"] for p in parameters]

X_train = pd.DataFrame(columns=param_names)
y_train = pd.Series(name="rmse")

def evaluation_function(param_dict):
    X_train.append(param_dict)
    x1 = param_dict["x1"]
    x2 = param_dict["x2"]
    mean = branin(x1, x2)
    y_train.append(mean)
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    sem = 0.0  # set to None to infer noise instead of assuming noiseless
    result = (mean, sem)
    return result

best_parameters, values, experiment, model = optimize(
    parameters=parameters, evaluation_function=evaluation_function, minimize=True,
)
