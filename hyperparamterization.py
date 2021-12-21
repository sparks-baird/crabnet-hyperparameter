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

init_notebook_plotting()

torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
train_loader, valid_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

from crabnet.model import data
train_df, val_df = data(elasticity, "train.csv")
train_df = train_df[:100]

from crabnet.train_crabnet import get_model

train_true, train_pred, formulas, train_sigma = crabnet_model.predict(train_df)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(train_true, train_pred)

def mse_error(parameterization):
    crabnet_model = get_model(
    mat_prop="elasticity",
    train_df=train_df,
    learningcurve=False,
    force_cpu=False,
    parameters = parameterization
    )
    train_true, train_pred, formulas, train_sigma = crabnet_model.predict(train_df)
    mse = mean_squared_error(train_true, train_pred)
    return mse

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=mse_error,
    objective_name='error',
    minimize = True
)

