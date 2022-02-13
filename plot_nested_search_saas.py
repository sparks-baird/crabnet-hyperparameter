from pprint import pprint
from matplotlib.pyplot import savefig

import plotly.graph_objects as go
import plotly.express as px
from ax.plot.marginal_effects import plot_marginal_effects
from ax.plot.feature_importances import (
    plot_feature_importance_by_feature_plotly,
    plot_feature_importance_by_metric_plotly,
)
from ax.plot.parallel_coordinates import (
    prepare_experiment_for_plotting,
    plot_parallel_coordinates_plotly,
)
from ax.plot.trace import optimization_trace_single_method
from ax.plot.slice import plot_slice_plotly, interact_slice_plotly
from sklearn.metrics import mean_absolute_error

from hyperparameterization import matplotlibify

total_trials = n_sobol + n_saas
for i, fold in enumerate(task.folds):
    trials = exp.trials.values()

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
    figpath = join(figure_dir, "best_objective_plot")
    fig.write_html(figpath + ".html")
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
    fig.write_image(figpath + ".png")

    fig = plot_feature_importance_by_feature_plotly(saas)
    fig.show()
    figpath = join(figure_dir, "plot_feature_importance_by_feature_plotly")
    fig.export_html(figpath + ".html")
    fig.write_image(figpath + ".png")

    fig = plot_feature_importance_by_metric_plotly(saas)
    fig.show()
    figpath = join(figure_dir, "plot_feature_importance_by_metric_plotly")
    fig.export_html(figpath + ".html")
    fig.write_image(figpath + ".png")

    fig = plot_marginal_effects(saas, metric)
    data = fig[0]["data"]
    layout = fig[0]["layout"]
    fig = go.Figure({"data": data, "layout": layout})
    fig.show()
    fig.export_html(join(figure_dir), "plot_feature_importance_by_feature_plotly.png")

    exp_df = prepare_experiment_for_plotting(exp)
    out_df = exp.fetch_data().df
    exp_df[metric] = out_df["mean"].values

    # need to install stats_model for trendlines, see
    # https://www.statsmodels.org/stable/install.html
    # 1D parameter projections
    fig = px.scatter(
        exp_df,
        x="batch_size",
        y=metric,
        trendline="lowess",
        trendline_options=dict(frac=0.25),
        trendline_scope="overall",
    )
    fig.show()
    fig, scale = matplotlibify(fig)
    fig.write_image(join(figure_dir, ".png"))

    fig = plot_parallel_coordinates_plotly(exp)
    fig.show()
    fig, scale = matplotlibify(fig)
    fig.write_image(join(figure_dir, ".png"))

    fig = interact_slice_plotly(saas)
    fig.show()
    fig, scale = matplotlibify(fig)
    fig.write_image(join(figure_dir, ".png"))
