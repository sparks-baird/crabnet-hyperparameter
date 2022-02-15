# %% imports
# NOTE: `pip install pyro-ppl` to use FULLYBAYESIAN (SAASBO)
from os.path import join

import crabnet
from submitit import AutoExecutor

from utils.matbench import matbench_fold, mb, task, figure_dir

# %% setup
# maes = []


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

# concatenation
for i, fold in enumerate(task.folds):
    test_pred, best_parameterization = jobs[i].result()
    task.record(fold, test_pred, best_parameterization)

my_metadata = {"algorithm_version": crabnet.__version__}
mb.add_metadata(my_metadata)
mb.to_file(join(figure_dir, "expt_gap_benchmark.json.gz"))
1 + 1

