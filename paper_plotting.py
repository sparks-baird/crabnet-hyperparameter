"""Reproduce paper figures for crabnet-hyperparameter."""
# %% load data
# default
from os import path
import pprint
from ax import Models
import numpy as np
import pandas as pd
from matbench import MatbenchBenchmark
from ax.storage.json_store.load import load_experiment
import plotly.graph_objects as go
from ax.plot.trace import optimization_trace_single_method
from ax.storage.metric_registry import register_metric
import torch
from utils.metrics import CrabNetMetric
from utils.plotting import matplotlibify, my_plot_feature_importance_by_feature_plotly
from ax.modelbridge.factory import get_GPEI
from ax.plot.scatter import plot_objective_vs_constraints
from ax.plot.feature_importances import (
    plot_feature_importance_by_feature_plotly,
    plot_feature_importance_by_metric_plotly,
)
from ax.plot.slice import plot_slice_plotly, interact_slice_plotly
from ax.core import ObservationFeatures

from ax.utils.notebook.plotting import render

from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly

from ax.plot.marginal_effects import plot_marginal_effects

from plotly import offline

dummy = False

fpaths = [
    "results/default/expt_gap_benchmark.json.gz",
    "results/ax/expt_gap_benchmark.json.gz",
    "results/saas/expt_gap_benchmark.json.gz",
    "results/gpei_10_90/expt_gap_benchmark.json.gz",
]

tmp_mb = MatbenchBenchmark()

mbs = [tmp_mb.from_file(fpath) for fpath in fpaths]
tasks = [list(mb.tasks)[0] for mb in mbs]
results = [task.results for task in tasks]
maes = [[result[f"fold_{i}"]["scores"]["mae"] for i in range(5)] for result in results]
params = [[result[f"fold_{i}"]["parameters"] for i in range(5)] for result in results]
dfs = [pd.DataFrame.from_records(param).T for param in params]

# fixup inconsistencies
default_df, ax_df, saas_df = dfs[0][0], dfs[1], dfs[2]
default_df = default_df.to_frame(name="default")
default_df.loc["batch_size", "default"] = 256
default_df.loc["criterion", "default"] = "RobustL1"
default_df = default_df.drop(["adam", "out_dims", "max_lr", "base_lr", "min_trust"])
ax_df.columns = [f"ax_fold_{i}" for i in range(5)]
saas_df.columns = [f"saas_fold_{i}" for i in range(5)]

df = pd.concat((default_df, ax_df, saas_df), axis=1)

default_mae, ax_mae, saas_mae = np.mean(maes[0]), maes[1], maes[2]
mae_df = pd.DataFrame(
    [default_mae] + ax_mae + saas_mae, columns=["mae"], index=df.columns
).T


def get_first(x):
    return x[0]


def get_second(x):
    return x[1]


df.loc["out_hidden_0"] = df.loc["out_hidden"].apply(get_first)
df = df.drop(index=["out_hidden"])

df.loc["betas_0"] = df.loc["betas"].apply(get_first)
df.loc["betas_1"] = df.loc["betas"].apply(get_second)
df = df.drop(index=["betas"])


def my_round(x):
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if np.round(x, 4) == 0:
            return np.format_float_scientific(x, precision=2, unique=False, trim="k")
        else:
            return np.format_float_positional(
                x, precision=4, unique=False, fractional=False, trim="k"
            )
    if isinstance(x, list):
        if isinstance(x[0], int):
            return x
        else:
            if np.round(x[0], 4) == 0:
                return [
                    np.format_float_scientific(a, precision=2, unique=False, trim=".")
                    for a in x
                ]
            else:
                return [
                    np.format_float_positional(
                        a, precision=4, unique=False, fractional=False, trim="."
                    )
                    for a in x
                ]


df = pd.concat((df, mae_df), axis=0)


print_df = df.applymap(my_round)

print_df.to_csv("results/parameters.csv")

ax_maes = df.loc["mae"][[f"ax_fold_{i}" for i in range(5)]]
saas_maes = df.loc["mae"][[f"saas_fold_{i}" for i in range(5)]]

if dummy:
    num_samples = 16
    warmup_steps = 32
else:
    num_samples = 256
    warmup_steps = 512

n_sobol = 10
n_saas = max(100 - n_sobol, 0)
total_trials = n_sobol + n_saas

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

torch.manual_seed(12345)  # To always get the same Sobol points


register_metric(metric_cls=CrabNetMetric)


def plot_and_save(fig_path, fig, mpl_kwargs={}, show=False):
    if show:
        offline.plot(fig)
    fig.write_html(fig_path + ".html")
    fig.to_json(fig_path + ".json")
    fig, scale = matplotlibify(fig, **mpl_kwargs)
    fig.write_image(fig_path + ".png")


non_range_names = ["criterion", "elem_prop", "bias"]

# %% Ax
figure_dir = "figures/ax"
# if dummy: # dummy no effect for Ax, only effect for SAASBO
#     figure_dir = path.join(figure_dir, "dummy")
fpaths = [path.join("experiments/ax", f"experiment{i}.json") for i in range(5)]
exps = [load_experiment(fpath) for fpath in fpaths]
metric = "mae"
ax_feature_importances = []
for i, (experiment, test_mae) in enumerate(zip(exps, ax_maes)):
    trials = experiment.trials.values()

    best_objectives = np.array([[trial.objective_mean for trial in trials]])

    parameter_strs = [
        pprint.pformat(trial.arm.parameters).replace("\n", "<br>") for trial in trials
    ]

    best_objective_plot = optimization_trace_single_method(
        y=best_objectives,
        optimization_direction="minimize",
        ylabel="Ax CrabNet MAE (eV)",
        hover_labels=parameter_strs,
        plot_trial_points=True,
    )

    fig_path = path.join(figure_dir, "best_objective_plot_" + str(i))

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

    # offline.plot(fig)
    fig.write_html(fig_path + ".html")
    fig.to_json(fig_path + ".json")
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
    fig.write_image(fig_path + ".png")

    fpath = path.join("experiments", "experiments0.json")

    model = get_GPEI(experiment, experiment.fetch_data())

    fig_path = path.join(figure_dir, "feature_importances_" + str(i))
    ax_feature_importances.append(model.feature_importances(metric))
    fig = plot_feature_importance_by_feature_plotly(model)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)

    if not dummy:

        fig_path = path.join(figure_dir, "marginal_effects_" + str(i))
        fig = plot_marginal_effects(model, metric)
        data, layout = [fig[0][nm] for nm in ["data", "layout"]]
        fig = go.Figure({"data": data, "layout": layout})
        # fig.update_yaxes(title_text="Percent worse than experimental average")
        # https://stackoverflow.com/a/63586646/13697228
        fig.layout["yaxis"].update(title_text="Percent worse than experimental average")
        fig.update_layout(title_text="")
        plot_and_save(
            fig_path,
            fig,
            mpl_kwargs=dict(size=18, height_inches=5.0, width_inches=15),
            show=False,
        )

        fig_path = path.join(figure_dir, "cross_validate_" + str(i))
        cv = cross_validate(model)
        fig = interact_cross_validation_plotly(cv)
        fig.update_xaxes(title_text="Actual MAE (eV)")
        fig.update_yaxes(title_text="Predicted MAE (eV)")
        plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

        fig_path = path.join(figure_dir, "slice", "slice_" + str(i))
        param_names = list(experiment.parameters.keys())

        for name in param_names:
            if name in non_range_names:
                continue  # skip categorical variables
            fig = plot_slice_plotly(model, name, "mae")
            fig.update_layout(title_text="")
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            plot_and_save(
                fig_path + f"_{name}",
                fig,
                mpl_kwargs=dict(width_inches=1.68, height_inches=1.68),
                show=False,
            )

    fig_path = path.join(figure_dir, "interact_slice_" + str(i))
    fig = interact_slice_plotly(model)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

fig_path = path.join(figure_dir, "avg_feature_importances")
feat_df = pd.DataFrame(ax_feature_importances).T
feat_df["mean"] = feat_df.mean(axis=1)
feat_df["std"] = feat_df.std(axis=1)
avg_ax_importances = feat_df["mean"].to_dict()
std_ax_importances = feat_df["std"].to_dict()
fig = my_plot_feature_importance_by_feature_plotly(
    model=None,
    feature_importances=avg_ax_importances,
    error_x=std_ax_importances,
    metric_names=["mae"],
)
plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)

# %% SAASBO
figure_dir = "figures/saas/sobol_10-saas_90"
if dummy:
    figure_dir = path.join(figure_dir, "dummy")
fpaths = [
    path.join("experiments/saas/sobol_10-saas_90", f"experiment{i}.json")
    for i in range(5)
]
exps = [load_experiment(fpath) for fpath in fpaths]
metric = "crabnet_mae"
saas_feature_importances = []
for i, (experiment, test_mae) in enumerate(zip(exps, saas_maes)):
    trials = experiment.trials.values()

    best_objectives = np.array([[trial.objective_mean for trial in trials]])

    parameter_strs = [
        pprint.pformat(trial.arm.parameters).replace("\n", "<br>") for trial in trials
    ]

    best_objective_plot = optimization_trace_single_method(
        y=best_objectives,
        optimization_direction="minimize",
        ylabel="Ax/SAASBO CrabNet MAE (eV)",
        hover_labels=parameter_strs,
        plot_trial_points=True,
    )
    # render(best_objective_plot)
    # plot_html = plot_config_to_html(best_objective_plot)

    fig_path = path.join(figure_dir, "best_objective_plot_" + str(i))

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

    # offline.plot(fig)
    fig.write_html(fig_path + ".html")
    fig.to_json(fig_path + ".json")
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
    fig.write_image(fig_path + ".png")

    fpath = path.join("experiments", "experiments0.json")

    saas = Models.FULLYBAYESIAN(
        experiment=experiment,
        data=experiment.fetch_data(),
        num_samples=num_samples,  # Increasing this may result in better model fits
        warmup_steps=warmup_steps,  # Increasing this may result in better model fits
        gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
        torch_device=tkwargs["device"],
        torch_dtype=tkwargs["dtype"],
        verbose=False,  # Set to True to print stats from MCMC
        disable_progbar=False,  # Set to False to print a progress bar from MCMC
    )

    fig_path = path.join(figure_dir, "feature_importances_" + str(i))
    saas_feature_importances.append(saas.feature_importances(metric))
    fig = plot_feature_importance_by_feature_plotly(saas)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)

    # not functional with SAAS apparently
    # fig_path = path.join(figure_dir, "marginal_effects_" + str(i))
    # fig = plot_marginal_effects(saas, metric)
    # data, layout = [fig[0][nm] for nm in ["data", "layout"]]
    # fig = go.Figure({"data": data, "layout": layout})
    # plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=15))

    fig_path = path.join(figure_dir, "cross_validate_" + str(i))
    cv = cross_validate(saas)
    fig = interact_cross_validation_plotly(cv)
    fig.update_xaxes(title_text="Actual MAE (eV)")
    fig.update_yaxes(title_text="Predicted MAE (eV)")
    plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

    fig_path = path.join(figure_dir, "slice", "slice_" + str(i))
    param_names = experiment.parameters.keys()

    if not dummy:
        for name in param_names:
            # skip categorical variables
            if name in non_range_names:
                continue
            # don't skip range parameters
            fig = plot_slice_plotly(saas, name, "crabnet_mae")
            fig.update_layout(title_text="")
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            plot_and_save(
                fig_path + f"_{name}",
                fig,
                mpl_kwargs=dict(width_inches=1.68, height_inches=1.68),  # for 5x5 grid
                show=False,
            )

    fig_path = path.join(figure_dir, "interact_slice_" + str(i))
    fig = interact_slice_plotly(saas)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

fig_path = path.join(figure_dir, "avg_feature_importances")
feat_df = pd.DataFrame(saas_feature_importances).T
feat_df["mean"] = feat_df.mean(axis=1)
feat_df["std"] = feat_df.std(axis=1)
avg_saas_importances = feat_df["mean"].to_dict()
std_saas_importances = feat_df["std"].to_dict()
fig = my_plot_feature_importance_by_feature_plotly(
    model=None,
    feature_importances=avg_saas_importances,
    error_x=std_saas_importances,
    metric_names=["crabnet_mae"],
)
plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)


# %% GPEI with 10 Sobol and 90 Bayes

figure_dir = "figures/gpei_10_90"
if dummy:
    figure_dir = path.join(figure_dir, "dummy")
fpaths = [
    path.join("experiments/sobol_10-gpei_90", f"experiment{i}.json") for i in range(5)
]
exps = [load_experiment(fpath) for fpath in fpaths]
metric = "crabnet_mae"
saas_feature_importances = []
for i, (experiment, test_mae) in enumerate(zip(exps, saas_maes)):
    trials = experiment.trials.values()

    best_objectives = np.array([[trial.objective_mean for trial in trials]])

    parameter_strs = [
        pprint.pformat(trial.arm.parameters).replace("\n", "<br>") for trial in trials
    ]

    best_objective_plot = optimization_trace_single_method(
        y=best_objectives,
        optimization_direction="minimize",
        ylabel="GPEI (10/90) CrabNet MAE (eV)",
        hover_labels=parameter_strs,
        plot_trial_points=True,
    )
    # render(best_objective_plot)
    # plot_html = plot_config_to_html(best_objective_plot)

    fig_path = path.join(figure_dir, "best_objective_plot_" + str(i))

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

    # offline.plot(fig)
    fig.write_html(fig_path + ".html")
    fig.to_json(fig_path + ".json")
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
    fig.write_image(fig_path + ".png")

    fpath = path.join("experiments", "experiments0.json")

    saas = Models.FULLYBAYESIAN(
        experiment=experiment,
        data=experiment.fetch_data(),
        num_samples=num_samples,  # Increasing this may result in better model fits
        warmup_steps=warmup_steps,  # Increasing this may result in better model fits
        gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
        torch_device=tkwargs["device"],
        torch_dtype=tkwargs["dtype"],
        verbose=False,  # Set to True to print stats from MCMC
        disable_progbar=False,  # Set to False to print a progress bar from MCMC
    )

    fig_path = path.join(figure_dir, "feature_importances_" + str(i))
    saas_feature_importances.append(saas.feature_importances(metric))
    fig = plot_feature_importance_by_feature_plotly(saas)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)

    # not functional with SAAS apparently
    # fig_path = path.join(figure_dir, "marginal_effects_" + str(i))
    # fig = plot_marginal_effects(saas, metric)
    # data, layout = [fig[0][nm] for nm in ["data", "layout"]]
    # fig = go.Figure({"data": data, "layout": layout})
    # plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=15))

    fig_path = path.join(figure_dir, "cross_validate_" + str(i))
    cv = cross_validate(saas)
    fig = interact_cross_validation_plotly(cv)
    fig.update_xaxes(title_text="Actual MAE (eV)")
    fig.update_yaxes(title_text="Predicted MAE (eV)")
    plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

    fig_path = path.join(figure_dir, "slice", "slice_" + str(i))
    param_names = experiment.parameters.keys()

    if not dummy:
        for name in param_names:
            # skip categorical variables
            if name in non_range_names:
                continue
            # don't skip range parameters
            fig = plot_slice_plotly(saas, name, "crabnet_mae")
            fig.update_layout(title_text="")
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            plot_and_save(
                fig_path + f"_{name}",
                fig,
                mpl_kwargs=dict(width_inches=1.68, height_inches=1.68),  # for 5x5 grid
                show=False,
            )

    fig_path = path.join(figure_dir, "interact_slice_" + str(i))
    fig = interact_slice_plotly(saas)
    plot_and_save(fig_path, fig, mpl_kwargs=dict(width_inches=4.0), show=False)

fig_path = path.join(figure_dir, "avg_feature_importances")
feat_df = pd.DataFrame(saas_feature_importances).T
feat_df["mean"] = feat_df.mean(axis=1)
feat_df["std"] = feat_df.std(axis=1)
avg_saas_importances = feat_df["mean"].to_dict()
std_saas_importances = feat_df["std"].to_dict()
fig = my_plot_feature_importance_by_feature_plotly(
    model=None,
    feature_importances=avg_saas_importances,
    error_x=std_saas_importances,
    metric_names=["crabnet_mae"],
)
plot_and_save(fig_path, fig, mpl_kwargs=dict(size=12), show=False)

1 + 1

# %% Code Graveyard
# fig = plot_objective_vs_constraints(saas, "batch_size", rel=False)
# data, layout = [fig[0][nm] for nm in ["data", "layout"]]
# fig = go.Figure({"data": data, "layout": layout})
# offline.plot(fig)

# generator_run = saas.gen(1)
# best_arm, _ = generator_run.best_arm_predictions
# best_params = best_arm.parameters

