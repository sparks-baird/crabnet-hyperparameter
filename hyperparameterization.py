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
from crabnet.utils.utils import RobustL1, RobustL2, BCEWithLogitsLoss

init_notebook_plotting()

train_df, val_df = data(elasticity)
train_df = train_df[:100]


def mse_error(parameterization):
    
    parameterization["out_hidden"] = [parameterization.get("out_hidden4")*8, parameterization.get("out_hidden4")*4, parameterization.get("out_hidden4")*2, parameterization.get("out_hidden4")]
    parameterization.pop("out_hidden4")
    
    parameterization["betas"] = (parameterization.get("betas1"), parameterization.get("betas2"))
    parameterization.pop("betas1")
    parameterization.pop("betas2")
    
    parameterization["d_model"] = parameterization["heads"] * round(parameterization["d_model"]/parameterization["heads"])
    
    parameterization["pos_scaler_log"] = 1 - parameterization["emb_scaler"] - parameterization["pos_scaler"]
    
    parameterization["epochs"] = parameterization["epochs_step"] * 4
    
    crabnet_model = get_model(
    mat_prop="elasticity",
    train_df=train_df,
    learningcurve=False,
    force_cpu=False,
  #  emb_scaler = parameterization.get("emb_scaler")
    **parameterization
    )
    train_true, train_pred, formulas, train_sigma = crabnet_model.predict(train_df)
    mse = mean_squared_error(train_true, train_pred)
    return {"error": mse}

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "batch_size", "type": "range", "bounds": [8, 512]},
        {"name": "fudge", "type": "range", "bounds": [0.0, 0.1]},
        {"name": "d_model", "type": "range", "bounds": [100, 1024]},
        {"name": "N", "type": "range", "bounds": [1, 10]},
        {"name": "heads", "type": "range", "bounds": [1, 10]},
        {"name": "out_hidden4", "type": "range", "bounds": [32, 512]},
        {"name": "emb_scaler", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "pos_scaler", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "bias", "type": "choice", "values": [False, True]},
        {"name": "dim_feedforward", "type": "range", "bounds": [1024, 4096]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "elem_prop", "type": "choice", "values": ["mat2vec", "jarvis", "magpie", "oliynyk", "onehot", "ptable"]},
        {"name": "epochs_step", "type": "range", "bounds": [5, 20]},
        {"name": "pe_resolution", "type": "range", "bounds": [2500, 10000]},
        {"name": "ple_resolution", "type": "range", "bounds": [2500, 10000]},
        {"name": "criterion", "type": "choice", "values": ["RobustL1", "RobustL2"]},
        {"name": "lr", "type": "range", "bounds": [0.0001, 0.006]},
        {"name": "betas1", "type": "range", "bounds": [0.5, 0.9999]},
        {"name": "betas2", "type": "range", "bounds": [0.5, 0.9999]},
        {"name": "eps", "type": "range", "bounds": [0.0000001, 0.0001]},
        {"name": "weight_decay", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "adam", "type": "choice", "values": [False, True]},
        {"name": "min_trust", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "k", "type": "range", "bounds": [1, 10]}
    ],
    experiment_name="hyperparameterization",
    evaluation_function=mse_error,
    objective_name='error',
    minimize = True,
    parameter_constraints=["betas1 <= betas2", "emb_scaler + pos_scaler <= 1"]
)

