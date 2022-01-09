from ax import RangeParameter, ParameterType
from ax.core import (
    SearchSpace,
    Experiment,
    OptimizationConfig,
    Objective,
    ObservationFeatures,
)
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.registry import Models
from ax.metrics import BraninMetric

branin_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)
exp = Experiment(
    name="test_branin",
    search_space=branin_search_space,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(name="branin", param_names=["x1", "x2"]), minimize=True,
        ),
    ),
    runner=SyntheticRunner(),
)

sobol = Models.SOBOL(exp.search_space)
for _ in range(5):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    trial.run()
    trial.mark_completed()

best_arm = None
for _ in range(15):
    gpei = Models.GPEI(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

fixed_features = ObservationFeatures({"x2": best_arm.parameters["x2"]})
for _ in range(15):
    gpei = Models.GPEI(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(
        1, search_space=branin_search_space, fixed_features=fixed_features,
    )
    best_arm2, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

exp.fetch_data()
best_parameters = best_arm2.parameters
1 + 1

