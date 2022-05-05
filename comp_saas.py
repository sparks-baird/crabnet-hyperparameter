# %% [markdown]
# ## Using non-linear inequality constraints in Ax
# This notebook comes with the following caveats:
# 1. The search space has to be [0, 1]^d
# 2. We need to pass in explicit `batch_initial_conditions` that satisfy the non-linear inequality constraints as starting points for optimizing the acquisition function.
# 3. BATCH_SIZE must be equal to 1.

# %%
from copy import copy
from os.path import join
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import torch

from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.nn.functional import normalize

from ax import (
    Data,
    Experiment,
    ParameterType,
    RangeParameter,
    SearchSpace,
    SumConstraint,
)

from ax.storage.json_store.save import save_experiment

# %%
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from utils.extraordinary import extraordinary_probability

from utils.metrics import CrabNetMetric
from utils.search import search_space

# from ax.utils.measurement.synthetic_functions import Hartmann6
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from torch.quasirandom import SobolEngine

from utils.sobol_candidates import nchoosek_sobol

dummy = False

result_dir = "results"
Path(result_dir).mkdir(exist_ok=True)

noise_sd = 0.1
synth_dither = 0.1
sem = None

d = 5  # HARD-CODED PARAMETER, i.e. 5 + 1 = 6 for Hartmann6Metric
param_names = [f"x{i}" for i in range(d + 1)]
subparam_names = param_names[:-1]  # sub-parameter names (i.e. all but last component)
params = [
    RangeParameter(
        name=parameter_name, parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0,
    )
    for parameter_name in subparam_names
]

metric = CrabNetMetric(name="objective")
optimization_config = OptimizationConfig(
    objective=Objective(metric=metric, minimize=True,)
)

# %% Let's see how we do via a brute force search
if dummy:
    comb_m = 10
else:
    comb_m = 18
candidates = nchoosek_sobol(
    param_names, n_slots=3, comb_m=comb_m, fixed_compositions=False
)
print(f"{len(candidates)} SOBOL candidates generated")
# compute the dither all at once, and add it to hartmann6 to get "true" fn
dither = metric.interp(candidates)
noise_free = metric.f_without_dither
ys = [noise_free(x) for x in candidates.values[:, :5]]
ys = np.array(ys) + dither
idx = np.argmin(ys)
print(f"minimum estimated via SOBOL search with true values: {ys[idx]:.4f}")
x_opt = candidates.iloc[idx]

# probability of finding a candidate within some percent of the estimated optimum
ys_noise = ys + noise_sd * np.random.randn(len(ys))
# for seemingly extraordinary candidates, do repeats to verify (i.e. with true values)
# mn = min(ys)
# mx = max(ys)
mn = -1.484  # as estimated by SAASBO
print(f"minimum estimated previously by SAASBO: {mn:.3f}")
mx = 0.0
thresh = 0.10  # i.e. within 10% of optimum

extraordinary_probability(ys, ys_noise, mx=mx, mn=mn, thresh=thresh)

# %% [markdown]
# We want to optimize $f_{\text{hartmann6}}(x)$ subject to an additional constraint $|| x ||_0 <= 3$.
#
# This constraint isn't differentiable, but it can be approximated by a differentiable relaxation using a sum of narrow Gaussian basis functions.
# Given a univariate Gaussian basis function $g_{\ell}(x)$ centered at zero with $\ell > 0$ small,
# we can approximate the constraint by: $|| x ||_0 \approx 6 - \sum_{i=1}^6 g_{\ell}(x_i) \leq 3$, which reduces to $\sum_{i=1}^6 g_{\ell}(x_i) \geq 3$.

# %%
def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)


def ineq_constraint(x, ell=1e-3):
    # Approximation of || x ||_0 <= 3. The constraint is >= 0 to conform with SLSQP
    return narrow_gaussian(x, ell).sum(dim=-1) - 3


# %% [markdown]
# ## BO-loop

# %%
def get_batch_initial_conditions(n, X, Y, raw_samples):
    """Generate starting points for the acquisition function optimization."""
    # 1. Draw `raw_samples` Sobol points and randomly set three parameters to zero to satisfy the constraint
    X_cand = SobolEngine(dimension=d, scramble=True).draw(raw_samples)
    X_cand = normalize(X_cand).to(torch.double)
    inds = torch.argsort(torch.rand(raw_samples, d), dim=-1)[:, :3]
    X_cand[torch.arange(X_cand.shape[0]).unsqueeze(-1), inds] = 0

    # 2. Fit a GP to the observed data, the right thing to do is to use the Ax model here
    gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # 3. Use EI to select the best points. Ideally, we should use the Ax acquisition function here as well
    EI = ExpectedImprovement(model=gp, best_f=Y.min(), maximize=False)
    X_cand = X_cand.unsqueeze(1)
    acq_vals = EI(X_cand)
    return X_cand[acq_vals.topk(n).indices]


# %%
BATCH_SIZE = 1
if dummy:
    N_INIT = 5
    N_BATCHES = 2
else:
    N_INIT = 10
    N_BATCHES = 90
print(f"Doing {N_INIT + N_BATCHES * BATCH_SIZE} evaluations")

# %%
# Experiment
experiment = Experiment(
    name="saasbo_experiment",
    search_space=search_space,
    optimization_config=optimization_config,
    runner=SyntheticRunner(),
)

# Initial Sobol points (set three random parameters to zero)
sobol = Models.SOBOL(search_space=experiment.search_space)
for _ in range(N_INIT):
    trial = sobol.gen(1)
    keys = copy(subparam_names)
    random.shuffle(keys)
    for k in keys[:3]:
        trial.arms[0]._parameters[k] = 0.0
    experiment.new_trial(trial).run()

# Run SAASBO
data = experiment.fetch_data()
for i in range(N_BATCHES):
    model = Models.FULLYBAYESIAN(
        experiment=experiment,
        data=data,
        num_samples=256,  # Increasing this may result in better model fits
        warmup_steps=512,  # Increasing this may result in better model fits
        gp_kernel="matern",  # "rbf" is the default in the paper, but we also support "matern"
        torch_dtype=torch.double,
        verbose=False,  # Set to True to print stats from MCMC
        disable_progbar=True,  # Set to False to print a progress bar from MCMC
    )
    batch_initial_conditions = get_batch_initial_conditions(
        n=20, X=model.model.Xs[0], Y=model.model.Ys[0], raw_samples=1024
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Filter SLSQP warnings
        generator_run = model.gen(
            BATCH_SIZE,
            model_gen_options={
                "optimizer_kwargs": {
                    "linear_constraints": [
                        (torch.arange(d), torch.ones(d), 1)
                    ],  # sum(x[:-1]) <= 1
                    "nonlinear_inequality_constraints": [ineq_constraint],
                    "batch_initial_conditions": batch_initial_conditions,
                }
            },
        )

    trial = experiment.new_batch_trial(generator_run=generator_run)
    for arm in trial.arms:
        arm._parameters = {k: 0.0 if v < 1e-3 else v for k, v in arm.parameters.items()}
        assert sum([v > 1e-3 for v in arm.parameters.values()]) <= 3
    trial.run()
    data = Data.from_multiple_data([data, trial.fetch_data()])

    fetched_data = trial.fetch_data()
    new_value = fetched_data.df["mean"].min()
    # best_value = fetched_data.true_df["mean"].min()
    best_value = data.df["mean"].min()

    arm_parameters = [arm.parameters for arm in list(experiment.arms_by_name.values())]
    arm_params = pd.DataFrame(arm_parameters).values
    y_true = np.array([metric.f(v) for v in arm_params])
    best_true_val = min(y_true)
    print(
        f"Iteration: {i}, Best in iteration {new_value:.3f}, ",
        f"Best so far: {best_value:.3f}, ",
        f"Best true so far: {best_true_val:.3f}",
    )

# %%
pd.options.display.float_format = "{:,.3f}".format
df = pd.DataFrame(arm_parameters)
df["x5"] = np.round(1 - df.values.sum(axis=1), decimals=6)
y_pred = data.df["mean"]
df["y_pred"] = y_pred
df["y_true"] = y_true
print(df)

# y_pred = df[]
extraordinary_probability(y_true, y_pred, mx=mx, mn=mn)

experiment_dir = result_dir
if dummy:
    experiment_dir = join("dummy", experiment_dir)
experiment_dir = join(
    experiment_dir,
    "experiments",
    f"{experiment.name}",
    f"N_INIT_{N_INIT}_BATCH_SIZE_{BATCH_SIZE}_N_BATCHES_{N_BATCHES}",
)
Path(experiment_dir).mkdir(exist_ok=True, parents=True)
experiment_fpath = join(experiment_dir, "experiment.json")
save_experiment(experiment, experiment_fpath)

df.to_csv(join(experiment_dir, "results.csv"))
