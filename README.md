# Bayesian Optimization of CrabNet Hyperparameters using Ax

We use [(my fork of) CrabNet](https://github.com/sgbaird/CrabNet) to adjust various hyperparameters for the experimental band gap matbench task (`matbench_expt_gap`). We chose this task because it is a composition-only dataset (CrabNet is a composition-only model) and because CrabNet is currently (2021-01-08) listed at the top of this leaderboard (with MODNet just marginally worse). In other words, when a model whose defaults already produce state-of-the-art property prediction performance, to what extent can it benefit from hyperparameter optimization (i.e. tuning parameters such as Neural Network dimensions, learning rates, etc.).

Eventually, I plan to incorporate this into (my fork of) CrabNet, but for now this can serve as an illustrative example of hyperparameter optimization using Bayesian adaptive design and could certainly be adapted to other models (e.g. Neural Networks), especially expensive-to-train models that have not undergone much by way of parameter tuning.

For more information on CrabNet's architecture, see [the original CrabNet paper published in Nature Partner Journals: Computational Materials](https://dx.doi.org/10.1038/s41524-021-00545-1).

See [`hyperparameterization.ipynb`](https://github.com/sparks-baird/crabnet-hyperparameter/blob/main/hyperparameterization.ipynb) for a more in-depth walkthrough of the process and results.

## Links
- [figures](figures)
- [Ax experiment JSON files](experiments)
- [requirements.txt](requirements.txt)
- [hyperparameterization.py](https://github.com/sparks-baird/crabnet-hyperparameter/blob/main/hyperparameterization.py) (`.py` script adapted into the Jupyter notebook mentioned above)
