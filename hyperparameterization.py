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

dummy = False
if dummy:
    n_splits = 2
    total_trials = 2
else:
    n_splits = 5
    total_trials = 100

# create dir https://stackoverflow.com/a/273227/13697228
experiment_dir = "experiments"
figure_dir = "figures"
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)


def matplotlibify(fig, size=24, width_inches=3.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=size, color="black")

    # app = QApplication(sys.argv)
    # screen = app.screens()[0]
    # dpi = screen.physicalDotsPerInch()
    # app.quit()

    fig.update_layout(
        font=font_dict,
        plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )
    fig.update(layout_coloraxis_showscale=False)

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale


def correct_parameterization(parameterization):
    pprint.pprint(parameterization)

    parameterization["out_hidden"] = [
        parameterization.get("out_hidden4") * 8,
        parameterization.get("out_hidden4") * 4,
        parameterization.get("out_hidden4") * 2,
        parameterization.get("out_hidden4"),
    ]
    parameterization.pop("out_hidden4")

    parameterization["betas"] = (
        parameterization.get("betas1"),
        parameterization.get("betas2"),
    )
    parameterization.pop("betas1")
    parameterization.pop("betas2")

    d_model = parameterization["d_model"]

    # make heads even (unless it's 1) (because d_model must be even)
    heads = parameterization["heads"]
    if np.mod(heads, 2) != 0:
        heads = heads + 1
    parameterization["heads"] = heads

    # NOTE: d_model must be divisible by heads
    d_model = parameterization["heads"] * round(d_model / parameterization["heads"])

    parameterization["d_model"] = d_model

    parameterization["pos_scaler_log"] = (
        1 - parameterization["emb_scaler"] - parameterization["pos_scaler"]
    )

    parameterization["epochs"] = parameterization["epochs_step"] * 4

    return parameterization


mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])
kf = KFold(n_splits=n_splits, shuffle=True, random_state=18012019)

task = list(mb.tasks)[0]
task.load()
for i, fold in enumerate(task.folds):
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:100]

    def crabnet_mae(parameterization):
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
        parameterization = correct_parameterization(parameterization)

        mae = 0.0
        for train_index, val_index in kf.split(train_val_df):
            train_df, val_df = (
                train_val_df.loc[train_index],
                train_val_df.loc[val_index],
            )
            crabnet_model = get_model(
                mat_prop="expt_gap",
                train_df=train_df,
                learningcurve=False,
                force_cpu=False,
                **parameterization
            )
            val_true, val_pred, val_formulas, val_sigma = crabnet_model.predict(val_df)
            # rmse = mean_squared_error(val_true, val_pred, squared=False)
            mae = mae + mean_absolute_error(val_true, val_pred)

            # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
            del crabnet_model
            gc.collect()
            torch.cuda.empty_cache()
        mae = mae / n_splits
        results = {"mae": mae}
        return results

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "batch_size", "type": "range", "bounds": [8, 256]},
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
        evaluation_function=crabnet_mae,
        objective_name="mae",
        minimize=True,
        parameter_constraints=["betas1 <= betas2", "emb_scaler + pos_scaler <= 1"],
        total_trials=total_trials,
    )
    print(best_parameters)
    print(values)

    experiment_fpath = join(experiment_dir, "experiment" + str(i) + ".json")
    save_experiment(experiment, experiment_fpath)

    # TODO: save plot, save experiment

    test_inputs = task.get_test_data(fold, include_target=False)

    test_df = pd.DataFrame(
        {"formula": test_inputs, "target": np.zeros_like(test_inputs)}
    )

    default_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
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
        **best_parameterization
    )
    # TODO: update CrabNet predict function to allow for no target specified
    test_true, test_pred, test_formulas, test_sigma = test_model.predict(test_df)
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    test_mae = mean_absolute_error(test_true, test_pred)

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
        ylabel="CrabNet RMSE (GPa)",
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
            name="test MAE",
            yaxis="y1",
        )
    )

    layout = best_objective_plot[0]["layout"]

    fig = go.Figure({"data": data, "layout": layout})

    fig.show()
    fig.write_html(figure_fpath + ".html")
    fig.to_json(figure_fpath + ".json")
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig, scale = matplotlibify(fig)
    fig.write_image(figure_fpath + ".png")

    task.record(fold, test_pred, params=best_parameterization)

my_metadata = {"algorithm_version": crabnet.__version__}

mb.add_metadata(my_metadata)

mb.to_file("expt_gap_benchmark.json.gz")


# train_df, val_df, test_df = data(elasticity, test_size=0.1)
# # train_df = train_df[:256]

# mdl = get_model(mat_prop="elasticity", train_df=train_df, learningcurve=False)
# v_true, v_pred, _, _ = mdl.predict(val_df)
# default = mean_squared_error(v_true, v_pred, squared=False)


# TODO: save results

1 + 1

# %% Code Graveyard
# parameterization = {
#     "batch_size": 480,
#     "fudge": 0.029900716710835697,
#     "d_model": 1020,
#     "N": 1,
#     "heads": 3,
#     "out_hidden4": 457,
#     "emb_scaler": 0.21383277047425508,
#     "pos_scaler": 0.1618593344464898,
#     "bias": False,
#     "dim_feedforward": 2161,
#     "dropout": 0.5495160408318043,
#     "epochs_step": 10,
#     "pe_resolution": 7475,
#     "ple_resolution": 6881,
#     "lr": 0.0006154754437506199,
#     "betas1": 0.5752653560178355,
#     "betas2": 0.8672460543309338,
#     "eps": 4.525619963891804e-05,
#     "weight_decay": 0.5064806314185262,
#     "adam": True,
#     "min_trust": 0.8777030305936933,
#     "alpha": 0.4261715365573764,
#     "k": 2,
#     "elem_prop": "onehot",
#     "criterion": "RobustL1",
# }

# parameterization = {
#     # "batch_size": 170,
#     "batch_size": None,
#     # "fudge": 0.013474978040903807,
#     # "d_model": 167,
#     "d_model": 512,
#     # "N": 4,
#     "N": 3,
#     # "heads": 6,
#     "heads": 4,
#     # "out_hidden4": 444,
#     "out_hidden4": 128,
#     # "emb_scaler": 0.0053000785410404205,
#     "emb_scaler": 0.33,
#     # "pos_scaler": 0.69534354750067,
#     "pos_scaler": 0.33,
#     "bias": False,
#     # "dim_feedforward": 4003,
#     # "dropout": 0.7298914920538664,
#     # "epochs_step": 19,
#     "epochs_step": 10,
#     # "pe_resolution": 3871,
#     # "ple_resolution": 4048,
#     # "lr": 0.0027536404109559953,
#     # "betas1": 0.8141271455351263,
#     "betas1": 0.9,
#     # "betas2": 0.8428701813892461,
#     "betas2": 0.999,
#     # "eps": 6.645583540787921e-05,
#     # "weight_decay": 0.4293977050110698,
#     # "adam": True,
#     # "min_trust": 0.5816364008933306,
#     # "alpha": 0.17873557657003403,
#     # "k": 8,
#     # "elem_prop": "mat2vec",
#     # "criterion": "RobustL2",
# }

# NOTE: adam and min_trust causing issues with onehot elem_prop
# parameterization = {
#     "batch_size": 170,
#     "fudge": 0.013474978040903807,
#     "d_model": 167,
#     "N": 4,
#     "heads": 6,
#     "out_hidden4": 444,
#     "emb_scaler": 0.0053000785410404205,
#     "pos_scaler": 0.69534354750067,
#     "bias": False,
#     "dim_feedforward": 4003,
#     "dropout": 0.7298914920538664,
#     "epochs_step": 19,
#     "pe_resolution": 3871,
#     "ple_resolution": 4048,
#     "lr": 0.0027536404109559953,
#     "betas1": 0.8141271455351263,
#     "betas2": 0.8428701813892461,
#     "eps": 6.645583540787921e-05,
#     "weight_decay": 0.4293977050110698,
#     # "adam": True,
#     # "min_trust": 0.5816364008933306,
#     "alpha": 0.17873557657003403,
#     "k": 8,
#     "elem_prop": "onehot",
#     "criterion": "RobustL2",
# }

# parameterization = {
#     "batch_size": 170,
#     "fudge": 0.013474978040903807,
#     "d_model": 167,
#     "N": 4,
#     "heads": 6,
#     "out_hidden4": 444,
#     "emb_scaler": 0.0053000785410404205,
#     "pos_scaler": 0.69534354750067,
#     "bias": False,
#     "dim_feedforward": 4003,
#     "dropout": 0.7298914920538664,
#     "epochs_step": 19,
#     "pe_resolution": 3871,
#     "ple_resolution": 4048,
#     "lr": 0.0027536404109559953,
#     "betas1": 0.8141271455351263,
#     "betas2": 0.8428701813892461,
#     "eps": 6.645583540787921e-05,
#     "weight_decay": 0.4293977050110698,
#     "min_trust": 0.5816364008933306,
#     "alpha": 0.17873557657003403,
#     "k": 8,
#     "elem_prop": "onehot",
#     "criterion": "RobustL2",
# }

# mse_error(parameterization)

# from crabnet.utils.utils import RobustL1, RobustL2, BCEWithLogitsLoss
# import pandas as pd
# import torch

# trial_strs = [str(trial) for trial in experiment.trials.values()]

# y=np.minimum.accumulate(best_objectives, axis=1),

# from ax.plot.render import plot_config_to_html
# from ax.utils.notebook.plotting import render
# from crabnet.data.materials_data import elasticity
# from crabnet.model import data

# mean_squared_error