"""Optimize CrabNet hyperparameters using Ax."""
import pprint
from os.path import join
from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go

import gc
import torch

from ax.storage.json_store.save import save_experiment
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import crabnet
from crabnet.train_crabnet import get_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from matbench.bench import MatbenchBenchmark

from utils.parameterization import correct_parameterization, crabnet_mae
from utils.plotting import matplotlibify

verbose = False
dummy = False
if dummy:
    n_splits = 2
    total_trials = 2
else:
    n_splits = 5
    total_trials = 1000

# create dir https://stackoverflow.com/a/273227/13697228
experiment_dir = "experiments"
figure_dir = "figures"
result_dir = "results"

if dummy:
    experiment_dir = join(experiment_dir, "dummy")
    figure_dir = join(figure_dir, "dummy")
    result_dir = join(result_dir, "dummy")

experiment_dir = join(experiment_dir, f"total_trials_{total_trials}")
figure_dir = join(figure_dir, f"total_trials_{total_trials}")
result_dir = join(result_dir, f"total_trials_{total_trials}")

Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)
Path(result_dir).mkdir(parents=True, exist_ok=True)

mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])
kf = KFold(n_splits=n_splits, shuffle=True, random_state=18012019)

task = list(mb.tasks)[0]
task.load()
for i, fold in enumerate(task.folds):
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    # TODO: treat train_val_df as Ax fixed_parameter
    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:100]

    def crabnet_mae_simple(parameterization):
        """Compute the mean absolute error of a CrabNet model.
        
        Assumes that `train_df` and `val_df` are predefined.

        Parameters
        ----------
        parameterization : dict
            Dictionary of the parameters passed to `get_model()` after some slight
            modification. 

        Returns
        -------
        results: dict
            Dictionary of `{"rmse": rmse}` where `rmse` is the root-mean-square error of the
            CrabNet model.
        """
        results = crabnet_mae(
            parameterization, train_val_df, n_splits=n_splits, kf=kf, verbose=verbose
        )
        return results

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "batch_size", "type": "range", "bounds": [32, 256]},
            {"name": "fudge", "type": "range", "bounds": [0.0, 0.1]},
            {"name": "d_model", "type": "range", "bounds": [100, 1024]},
            {"name": "N", "type": "range", "bounds": [1, 10]},
            {"name": "heads", "type": "range", "bounds": [1, 10]},
            {"name": "out_hidden4", "type": "range", "bounds": [32, 512]},
            {"name": "emb_scaler", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "pos_scaler", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "bias", "type": "choice", "values": [False, True]},
            {"name": "dim_feedforward", "type": "range", "bounds": [1024, 4096],},
            {"name": "dropout", "type": "range", "bounds": [0.0, 1.0]},
            # jarvis and oliynyk don't have enough elements
            # ptable contains str, which isn't a handled case
            {
                "name": "elem_prop",
                "type": "choice",
                "values": [
                    "mat2vec",
                    "magpie",
                    "onehot",
                ],  # "jarvis", "oliynyk", "ptable"
            },
            {"name": "epochs_step", "type": "range", "bounds": [5, 20]},
            {"name": "pe_resolution", "type": "range", "bounds": [2500, 10000]},
            {"name": "ple_resolution", "type": "range", "bounds": [2500, 10000],},
            {
                "name": "criterion",
                "type": "choice",
                "values": ["RobustL1", "RobustL2"],
            },
            {"name": "lr", "type": "range", "bounds": [0.0001, 0.006]},
            {"name": "betas1", "type": "range", "bounds": [0.5, 0.9999]},
            {"name": "betas2", "type": "range", "bounds": [0.5, 0.9999]},
            {"name": "eps", "type": "range", "bounds": [0.0000001, 0.0001]},
            {"name": "weight_decay", "type": "range", "bounds": [0.0, 1.0]},
            # {"name": "adam", "type": "choice", "values": [False, True]}, # issues with onehot
            # {"name": "min_trust", "type": "range", "bounds": [0.0, 1.0]}, #issues with onehot
            {"name": "alpha", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "k", "type": "range", "bounds": [2, 10]},
        ],
        experiment_name="crabnet-hyperparameter",
        evaluation_function=crabnet_mae_simple,
        objective_name="mae",
        minimize=True,
        parameter_constraints=["betas1 <= betas2", "emb_scaler + pos_scaler <= 1"],
        total_trials=total_trials,
    )
    print(best_parameters)
    print(values)

    experiment_fpath = join(experiment_dir, "experiment" + str(i) + ".json")
    save_experiment(experiment, experiment_fpath)

    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    test_df = pd.DataFrame({"formula": test_inputs, "target": test_outputs})

    default_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
        verbose=verbose,
    )

    default_true, default_pred, default_formulas, default_sigma = default_model.predict(
        test_df
    )
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    default_mae = mean_absolute_error(default_true, default_pred)

    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
    del default_model
    gc.collect()
    torch.cuda.empty_cache()

    best_parameterization = correct_parameterization(best_parameters)
    test_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
        verbose=verbose,
        **best_parameterization,
    )
    # TODO: update CrabNet predict function to allow for no target specified
    test_true, test_pred, test_formulas, test_sigma = test_model.predict(test_df)
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    test_mae = mean_absolute_error(test_outputs, test_pred)

    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
    del test_model
    gc.collect()
    torch.cuda.empty_cache()

    trials = experiment.trials.values()

    best_objectives = np.array([[trial.objective_mean for trial in trials]])

    parameter_strs = [
        pprint.pformat(trial.arm.parameters).replace("\n", "<br>") for trial in trials
    ]

    best_objective_plot = optimization_trace_single_method(
        y=best_objectives,
        optimization_direction="minimize",
        ylabel="MAE (eV)",
        hover_labels=parameter_strs,
        plot_trial_points=True,
    )
    # render(best_objective_plot)
    # plot_html = plot_config_to_html(best_objective_plot)

    figure_fpath = join(figure_dir, "best_objective_plot_" + str(i))
    # with open(figure_fpath, "w") as f:
    #     f.write(plot_html)

    data = best_objective_plot[0]["data"]

    data.append(
        go.Scatter(
            x=(1, total_trials),
            y=(default_mae, default_mae),
            mode="lines",
            line={"dash": "dash"},
            name="default MAE",
            yaxis="y1",
        )
    )

    data.append(
        go.Scatter(
            x=(1, total_trials),
            y=(test_mae, test_mae),
            mode="lines",
            line={"dash": "dash"},
            name="best model test MAE",
            yaxis="y1",
        )
    )

    layout = best_objective_plot[0]["layout"]

    fig = go.Figure({"data": data, "layout": layout})

    fig.show()
    fig.write_html(figure_fpath + ".html")
    fig.to_json(figure_fpath + ".json")
    fig.update_layout(
        legend=dict(
            font=dict(size=16),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0)",
        )
    )
    fig, scale = matplotlibify(fig)
    fig.write_image(figure_fpath + ".png")

    task.record(fold, test_pred, params=best_parameterization)

my_metadata = {"algorithm_version": crabnet.__version__}

mb.add_metadata(my_metadata)

mb.to_file(join(result_dir, "expt_gap_benchmark.json.gz"))

1 + 1

