# %% imports
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd

from matbench.bench import MatbenchBenchmark

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
from ax.plot.slice import plot_slice_plotly, interact_slice_plotly

import torch

from ax import RangeParameter, ChoiceParameter, ParameterType, Data
from ax.core import (
    SearchSpace,
    Metric,
    Experiment,
    OptimizationConfig,
    Objective,
    ObservationFeatures,
)
from ax.core.parameter_constraint import SumConstraint, OrderConstraint
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models

import crabnet

from utils.matbench import get_test_results
from utils.parameterization import crabnet_mae

# %% setup
dummy = True
metric = "crabnet_mae"

if dummy:
    n_splits = 2
else:
    n_splits = 5

# create dir https://stackoverflow.com/a/273227/13697228
experiment_dir = "experiments"
figure_dir = "figures"
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)

torch.manual_seed(12345)  # To always get the same Sobol points

# %% constraint parameters and constraints
betas1 = RangeParameter(
    name="betas1", parameter_type=ParameterType.FLOAT, lower=0.5, upper=0.9999
)
betas2 = RangeParameter(
    name="betas2", parameter_type=ParameterType.FLOAT, lower=0.5, upper=0.9999
)
emb_scaler = RangeParameter(
    name="emb_scaler", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
)
pos_scaler = RangeParameter(
    name="pos_scaler", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
)
order_constraint = OrderConstraint(lower_parameter=betas1, upper_parameter=betas2)
sum_constraint = SumConstraint(
    parameters=[emb_scaler, pos_scaler], is_upper_bound=True, bound=1.0
)
parameter_constraints = [order_constraint, sum_constraint]

# %% search space
search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="batch_size", parameter_type=ParameterType.INT, lower=32, upper=256
        ),
        RangeParameter(
            name="fudge", parameter_type=ParameterType.FLOAT, lower=0.0, upper=0.1
        ),
        RangeParameter(
            name="d_model", parameter_type=ParameterType.INT, lower=100, upper=1024
        ),
        RangeParameter(name="N", parameter_type=ParameterType.INT, lower=1, upper=10),
        RangeParameter(
            name="heads", parameter_type=ParameterType.INT, lower=1, upper=10
        ),
        RangeParameter(
            name="out_hidden4", parameter_type=ParameterType.INT, lower=32, upper=512
        ),
        emb_scaler,
        pos_scaler,
        ChoiceParameter(
            name="bias", parameter_type=ParameterType.BOOL, values=[False, True]
        ),
        RangeParameter(
            name="dim_feedforward",
            parameter_type=ParameterType.INT,
            lower=1024,
            upper=4096,
        ),
        RangeParameter(
            name="dropout", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        ),
        ChoiceParameter(
            name="elem_prop",
            parameter_type=ParameterType.STRING,
            values=["mat2vec", "magpie", "onehot"],
        ),
        RangeParameter(
            name="epochs_step", parameter_type=ParameterType.INT, lower=5, upper=20
        ),
        RangeParameter(
            name="pe_resolution",
            parameter_type=ParameterType.INT,
            lower=2500,
            upper=10000,
        ),
        RangeParameter(
            name="ple_resolution",
            parameter_type=ParameterType.INT,
            lower=2500,
            upper=10000,
        ),
        ChoiceParameter(
            name="criterion",
            parameter_type=ParameterType.STRING,
            values=["RobustL1", "RobustL2"],
        ),
        RangeParameter(
            name="lr", parameter_type=ParameterType.FLOAT, lower=0.0001, upper=0.006
        ),
        betas1,
        betas2,
        RangeParameter(
            name="eps",
            parameter_type=ParameterType.FLOAT,
            lower=0.0000001,
            upper=0.0001,
        ),
        RangeParameter(
            name="weight_decay",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
        RangeParameter(
            name="alpha", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0,
        ),
        RangeParameter(name="k", parameter_type=ParameterType.INT, lower=2, upper=10),
    ],
    parameter_constraints=parameter_constraints,
)

param_names = list(search_space.parameters.keys())
# %% CrabNetMetric
class CrabNetMetric(Metric):
    def __init__(self, name, train_val_df):
        self.train_val_df = train_val_df
        super().__init__(name=name)

    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters

            # TODO: add timing info as optional parameter and as outcome metric
            # TODO: maybe add interval score calculation as outcome metric
            mean = crabnet_mae(params, train_val_df=train_val_df, n_splits=n_splits)

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": mean,
                    "sem": None,
                }
            )
        return Data(df=pd.DataFrame.from_records(records))


# %% matbench loop
if dummy:
    n_sobol = 2
    n_gpei1 = 3
    n_gpei2 = 3
else:
    n_sobol = 2 * len(search_space.parameters)
    # n_gpei1 = max(100 - n_sobol, 0)
    n_gpei1 = 0
    n_gpei2 = max(100 - n_sobol, 0)  # 100

mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

task = list(mb.tasks)[0]
task.load()
for i, fold in enumerate(task.folds):
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:100]

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=CrabNetMetric(name=metric, train_val_df=train_val_df), minimize=True,
        ),
    )
    # TODO: use status_quo (Arm) as default CrabNet parameters
    exp = Experiment(
        name="nested_crabnet_mae",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    sobol = Models.SOBOL(exp.search_space)
    for _ in range(n_sobol):
        trial = exp.new_trial(generator_run=sobol.gen(1))
        trial.run()
        trial.mark_completed()

    best_arm1 = None
    for _ in range(n_gpei1):
        gpei = Models.GPEI(experiment=exp, data=exp.fetch_data())
        generator_run = gpei.gen(1)
        best_arm1, _ = generator_run.best_arm_predictions
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    n_param = len(param_names)
    n_gpei_per_param = n_gpei2 / n_param

    # initialize
    best_arm = best_arm1
    feature_importances = gpei.feature_importances(metric)
    # HACK: average the ChoiceParameter feature importances into a scalar
    # https://www.kite.com/python/answers/how-to-rename-a-dictionary-key-in-python
    feature_importances["elem_prop"] = np.mean(
        [
            feature_importances.pop("elem_prop_OH_PARAM__0"),
            feature_importances.pop("elem_prop_OH_PARAM__1"),
            feature_importances.pop("elem_prop_OH_PARAM__2"),
        ]
    )
    feature_importances["criterion"] = feature_importances.pop("criterion_OH_PARAM_")
    unfixed_importances = copy(feature_importances)
    fixed_params = {}
    ct = 0
    # recursive feature elimination of hyperparameters (RFE-h)
    for i in range(n_param):
        # key corresponding to min value https://stackoverflow.com/a/3282904/13697228
        least_important = min(unfixed_importances, key=unfixed_importances.get)
        fixed_params[least_important] = best_arm.parameters[least_important]
        fixed_features = ObservationFeatures(fixed_params)
        unfixed_importances.pop(least_important)
        # switch between ceil and floor for even and odd, resp.
        if i % 2 == 0:
            n_tmp = np.ceil(n_gpei_per_param)
        else:
            n_tmp = np.floor(n_gpei_per_param)
        n_tmp = int(max(n_tmp, 1))
        for _ in range(n_tmp):
            ct = ct + 1
            if ct <= n_gpei2:
                gpei2 = Models.GPEI(experiment=exp, data=exp.fetch_data())
                generator_run = gpei.gen(
                    1, search_space=search_space, fixed_features=fixed_features,
                )
                best_arm2, _ = generator_run.best_arm_predictions
                trial = exp.new_trial(generator_run=generator_run)
                trial.run()
                trial.mark_completed()
        best_arm = best_arm2
        # NOTE: feature_importances contains fixed features
        feature_importances = gpei.feature_importances(metric)

    exp.fetch_data()
    best_parameters = best_arm.parameters

    fig = plot_feature_importance_by_feature_plotly(gpei2)
    fig.show()

    fig = plot_feature_importance_by_metric_plotly(gpei2)
    fig.show()

    fig = plot_marginal_effects(gpei2, metric)
    data = fig[0]["data"]
    layout = fig[0]["layout"]
    fig = go.Figure({"data": data, "layout": layout})
    fig.show()

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

    fig = plot_parallel_coordinates_plotly(exp)
    fig.show()

    fig = interact_slice_plotly(gpei2)
    fig.show()

    test_pred, default_mae, test_mae, best_parameterization = get_test_results(
        task, fold, best_parameters, train_val_df
    )

    task.record(fold, test_pred, params=best_parameterization)

my_metadata = {"algorithm_version": crabnet.__version__}
mb.add_metadata(my_metadata)
mb.to_file("expt_gap_benchmark.json.gz")
1 + 1

# %% Code Graveyard
# min_importance = min(unfixed_importances.values())
# min_index = unfixed_importances.values().index(min_importance)
# least_important = unfixed_importances.keys[min_index]

# fixed_features = ObservationFeatures({"betas1": best_arm.parameters["betas1"]})
# for _ in range(n_gpei2):
#     gpei2 = Models.GPEI(experiment=exp, data=exp.fetch_data())
#     generator_run = gpei.gen(
#         1, search_space=search_space, fixed_features=fixed_features,
#     )
#     best_arm2, _ = generator_run.best_arm_predictions
#     trial = exp.new_trial(generator_run=generator_run)
#     trial.run()
#     trial.mark_completed()

# unfixed_importances = [
#     feature_importances.pop(fixed_name) for fixed_name in fixed_params.keys()
# ]

# table_view_plot(exp, exp.fetch_data())
# fig = plot_slice_plotly(gpei2, param_name="batch_size", metric_name="crabnet_mae")
# fig.show()

