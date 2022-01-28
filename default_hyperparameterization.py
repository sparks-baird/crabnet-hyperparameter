"""Optimize CrabNet hyperparameters using Ax."""
from os.path import join
from pathlib import Path
import pandas as pd

import gc
import torch

import crabnet
from crabnet.train_crabnet import get_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from matbench.bench import MatbenchBenchmark

dummy = False

# create dir https://stackoverflow.com/a/273227/13697228
experiment_dir = join("experiments", "default")
figure_dir = join("figures", "default")
result_dir = join("results", "default")
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
Path(figure_dir).mkdir(parents=True, exist_ok=True)
Path(result_dir).mkdir(parents=True, exist_ok=True)

mb = MatbenchBenchmark(autoload=False, subset=["matbench_expt_gap"])

default_maes = []
task = list(mb.tasks)[0]
task.load()
for i, fold in enumerate(task.folds):
    train_inputs, train_outputs = task.get_train_and_val_data(fold)

    train_val_df = pd.DataFrame(
        {"formula": train_inputs.values, "target": train_outputs.values}
    )
    if dummy:
        train_val_df = train_val_df[:100]

    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    test_df = pd.DataFrame({"formula": test_inputs, "target": test_outputs})

    default_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
    )

    default_true, default_pred, default_formulas, default_sigma = default_model.predict(
        test_df
    )

    default_mae = mean_absolute_error(default_true, default_pred)
    default_maes.append(default_mae)

    default_params = dict(
        fudge=0.02,
        d_model=512,
        out_dims=3,
        N=3,
        heads=4,
        out_hidden=[1024, 512, 256, 128],
        emb_scaler=1.0,
        pos_scaler=1.0,
        pos_scaler_log=1.0,
        bias=False,
        dim_feedforward=2048,
        dropout=0.1,
        elem_prop="mat2vec",
        pe_resolution=5000,
        ple_resolution=5000,
        epochs=40,
        epochs_step=10,
        criterion=None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
        min_trust=None,
        alpha=0.5,
        k=6,
        base_lr=1e-4,
        max_lr=6e-3,
    )

    task.record(fold, default_pred, params=default_params)

    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
    del default_model
    gc.collect()
    torch.cuda.empty_cache()

my_metadata = {"algorithm_version": crabnet.__version__}

mb.add_metadata(my_metadata)

mb.to_file(join(result_dir, "expt_gap_benchmark.json.gz"))

print(default_maes)
