# %% imports
# NOTE: `pip install pyro-ppl` to use FULLYBAYESIAN (SAASBO)
from os.path import join

import crabnet
from submitit import AutoExecutor

import cloudpickle as pickle

from utils.matbench import matbench_fold, mb, task, figure_dir

# %% submission
log_folder = "log_ax/%j"
walltime = 60
# partition, account = ["notchpeak-gpu", "notchpeak-gpu"]
partition, account = ["notchpeak-guest", "owner-guest"]
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_additional_parameters={"account": account},
)
jobs = executor.map_array(matbench_fold, task.folds)  # sbatch array
job_ids = [job.job_id for job in jobs]
# https://www.hpc2n.umu.se/documentation/batchsystem/job-dependencies
job_ids_str = ":".join(job_ids)  # e.g. "3937257_0:3937257_1:..."

with open("jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)

# %% collection
savepath = join(figure_dir, "expt_gap_benchmark.json.gz")


def collect_results():
    with open("jobs.pkl", "rb") as f:
        jobs = pickle.load(f)
    # concatenation
    for i, fold in enumerate(task.folds):
        test_pred, best_parameterization = jobs[i].result()
        task.record(fold, test_pred, best_parameterization)

    my_metadata = {"algorithm_version": crabnet.__version__}
    mb.add_metadata(my_metadata)
    mb.to_file(savepath)


collect_folder = "log_matbench/%j"
walltime = 10
collector = AutoExecutor(folder=collect_folder)
collector.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_additional_parameters={
        "account": account,
        "dependency": f"afterok:{job_ids_str}",
        "mail-type": "All",
    },
)
collector_job = collector.submit(collect_results)  # sbatch array

print(
    f"Waiting for submission jobs ({job_ids_str}) to complete before running collector job ({collector_job.job_id}). Feel free to exit and use the matbench output file that will be saved to {savepath} after all jobs have run."
)

