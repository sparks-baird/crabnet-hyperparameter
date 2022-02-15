# %% imports
# NOTE: `pip install pyro-ppl` to use FULLYBAYESIAN (SAASBO)
from time import time
from os.path import join
from pathlib import Path
import numpy as np

import pandas as pd

from matbench.bench import MatbenchBenchmark

import torch

from ax.storage.json_store.save import save_experiment
from ax import Data
from ax.core import (
    Experiment,
    OptimizationConfig,
    Objective,
)
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models

import crabnet

from utils.matbench import get_test_results
from utils.search import search_space
from utils.metrics import CrabNetMetric

from submitit import AutoExecutor

# %% setup
dummy = True
metric = "crabnet_mae"

if dummy:
    n_splits = 2
    n_sobol = 2
    n_saas = 3
    num_samples = 16
    warmup_steps = 32
else:
    n_splits = 5
    # n_sobol = 2 * len(search_space.parameters)
    n_sobol = 10
    n_saas = max(100 - n_sobol, 0)
    num_samples = 256
    warmup_steps = 512

torch.manual_seed(12345)  # To always get the same Sobol points
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# create dir https://stackoverflow.com/a/273227/13697228
parameter_str = join("saas", f"sobol_{n_sobol}-saas_{n_saas}")
experiment_dir = join("experiments", parameter_str)
figure_dir = join("figures", parameter_str)
if dummy:
    experiment_dir = join(experiment_dir, "dummy")
    figure_dir = join(figure_dir, "dummy")
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)


param_names = list(search_space.parameters.keys())
# %% matbench loop
mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

task = list(mb.tasks)[0]
task.load()
maes = []


def matbench_fold(fold):
    t0 = time()
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:25]

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=CrabNetMetric(name=metric, train_val_df=train_val_df), minimize=True,
        ),
    )
    # TODO: use status_quo (Arm) as default CrabNet parameters
    exp = Experiment(
        name="nested_crabnet_mae_saas",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    sobol = Models.SOBOL(exp.search_space)
    print("evaluating SOBOL points")
    for _ in range(n_sobol):
        print(_)
        trial = exp.new_trial(generator_run=sobol.gen(1))
        trial.run()
        trial.mark_completed()

    data = exp.fetch_data()
    j = -1
    new_value = np.nan
    best_so_far = np.nan
    for j in range(n_saas):
        saas = Models.FULLYBAYESIAN(
            experiment=exp,
            data=exp.fetch_data(),
            num_samples=num_samples,  # Increasing this may result in better model fits
            warmup_steps=warmup_steps,  # Increasing this may result in better model fits
            gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
            torch_device=tkwargs["device"],
            torch_dtype=tkwargs["dtype"],
            verbose=False,  # Set to True to print stats from MCMC
            disable_progbar=True,  # Set to False to print a progress bar from MCMC
        )
        generator_run = saas.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()
        data = Data.from_multiple_data([data, trial.fetch_data()])
        new_value = trial.fetch_data().df["mean"].min()
        best_so_far = data.df["mean"].min()
        tf = time()
        print(
            f"iter{j}, BestInIter:{new_value:.3f}, BestSoFar:{best_so_far:.3f} elapsed time: {tf - t0}",
        )

    exp.fetch_data()
    best_parameters = best_arm.parameters

    experiment_fpath = join(experiment_dir, "experiment" + str(i) + ".json")
    save_experiment(exp, experiment_fpath)

    test_pred, default_mae, test_mae, best_parameterization = get_test_results(
        task, fold, best_parameters, train_val_df
    )
    print(f"default_mae: {default_mae}")
    print(f"test_mae: {test_mae}")
    maes.append(test_mae)  # [0.32241879861870626, ...]

    task.record(fold, test_pred, params=best_parameterization)


log_folder = "log_test/%j"
walltime = 60
partition, account = ["lonepeak", "sparks"]
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_additional_parameters={"account": account},
)
jobs = executor.map_array(matbench_fold, task.folds)  # sbatch array

print(maes)
print(np.mean(maes))
my_metadata = {"algorithm_version": crabnet.__version__}
mb.add_metadata(my_metadata)
mb.to_file(join(figure_dir, "expt_gap_benchmark.json.gz"))
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

