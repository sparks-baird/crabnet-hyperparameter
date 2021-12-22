#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:20:48 2021

@author: marianneliu
"""

import torch
import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
from crabnet.train_crabnet import get_model
from crabnet.data.materials_data import elasticity
from crabnet.model import data
from sklearn.metrics import mean_squared_error

init_notebook_plotting()

train_df, val_df = data(elasticity)
train_df = train_df[:100]

def mse_error(parameterization):
    crabnet_model = get_model(
    mat_prop="elasticity",
    train_df=train_df,
    learningcurve=False,
    force_cpu=False,
    emb_scaler = parameterization.get("emb_scaler")
    )
    train_true, train_pred, formulas, train_sigma = crabnet_model.predict(train_df)
    mse = mean_squared_error(train_true, train_pred)
    return {"error": mse}

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "emb_scaler", "type": "range", "bounds": [0, 10]}
    ],
    experiment_name="hyperparameterization",
    evaluation_function=mse_error,
    objective_name='error',
    minimize = True
)

