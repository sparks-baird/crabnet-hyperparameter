# Bayesian Optimization of CrabNet Hyperparameters using Ax

[![DOI](https://img.shields.io/badge/Zenodo-10.5281/fzenodo.6355044-blue)](https://doi.org/10.5281/zenodo.6355044) [![DOI](https://img.shields.io/badge/ComMatSci-10.1016/j.commatsci.2022.111505-green)](https://doi.org/10.1016/j.commatsci.2022.111505) [![arXiv](https://img.shields.io/badge/arXiv-2203.12597-b31b1b.svg)](https://doi.org/10.48550/arXiv.2203.12597)

We use [Ax Bayesian optimization](https://ax.dev/docs/bayesopt.html) to adjust hyperparameters for [(my fork of) CrabNet](https://github.com/sgbaird/CrabNet) for the experimental band gap matbench task (`matbench_expt_gap`). We chose this task because it is a composition-only dataset (CrabNet is a composition-only model) and because CrabNet is currently (2021-01-08) listed at the top of this leaderboard (with MODNet just marginally worse). In other words, when a model whose defaults already produce state-of-the-art property prediction performance, to what extent can it benefit from hyperparameter optimization (i.e. tuning parameters such as Neural Network dimensions, learning rates, etc.)?

As of 2022-04-05, `Ax/SAASBO CrabNet v1.2.7` [holds the current per-task leaderboard](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/) on the `matbench_expt_gap` task ([c3b910e4f0](https://github.com/materialsproject/matbench/tree/c3b910e4f06b79eea1a8a6c7b67ea5a605948306)). For additional details, please consider reading the [CMS article](https://doi.org/10.1016/j.commatsci.2022.111505) or the [preprint](https://doi.org/10.48550/arXiv.2203.12597). For more information on CrabNet's architecture, see [the original CrabNet paper published in Nature Partner Journals: Computational Materials](https://dx.doi.org/10.1038/s41524-021-00545-1).

This case study can serve as an illustrative example of hyperparameter optimization using Bayesian adaptive design and could certainly be adapted to other models (e.g. Neural Networks), especially expensive-to-train models that have not undergone much by way of parameter tuning. See [`hyperparameterization.ipynb`](https://github.com/sparks-baird/crabnet-hyperparameter/blob/main/hyperparameterization.ipynb) for a more in-depth walkthrough of the process and results.

## Links
- [figures](figures)
- [Ax experiment JSON files](experiments)
- [requirements.txt](requirements.txt)
- [hyperparameterization.py](https://github.com/sparks-baird/crabnet-hyperparameter/blob/main/hyperparameterization.py) (`.py` script adapted into the Jupyter notebook mentioned above)

Eventually, I plan to incorporate this into (my fork of) CrabNet, but for now this can serve as an illustrative example of hyperparameter optimization using Bayesian adaptive design and could certainly be adapted to other models (e.g. Neural Networks), especially expensive-to-train models that have not undergone much by way of parameter tuning.

## Citing
If you find this useful, please consider citing:
> Baird, S. G.; Liu, M.; Sparks, T. D. High-Dimensional Bayesian Optimization of 23 Hyperparameters over 100 Iterations for an Attention-Based Network to Predict Materials Property: A Case Study on CrabNet Using Ax Platform and SAASBO. Computational Materials Science 2022, 211, 111505. https://doi.org/10.1016/j.commatsci.2022.111505.

In addition to the above manuscript citation, if you use this code, please also cite the following for all versions (alternatively, a [specific version](https://zenodo.org/badge/latestdoi/431324974)):
> Sterling Baird, sgbaird-alt, & mliu7051. (2022). sparks-baird/crabnet-hyperparameter. Zenodo. https://doi.org/10.5281/zenodo.6355044
