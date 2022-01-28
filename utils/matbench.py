import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
import gc
from utils.parameterization import correct_parameterization
from crabnet.train_crabnet import get_model


def get_test_results(task, fold, best_parameters, train_val_df):
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
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    default_mae = mean_absolute_error(default_true, default_pred)

    # deallocate CUDA memory https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/28
    del default_model
    gc.collect()
    torch.cuda.empty_cache()

    best_parameterization = correct_parameterization(best_parameters)
    test_model = get_model(
        mat_prop="expt_gap",
        train_df=train_val_df,
        learningcurve=False,
        force_cpu=False,
        **best_parameterization
    )
    # TODO: update CrabNet predict function to allow for no target specified
    test_true, test_pred, test_formulas, test_sigma = test_model.predict(test_df)
    # rmse = mean_squared_error(val_true, val_pred, squared=False)
    test_mae = mean_absolute_error(test_true, test_pred)

    return test_pred, default_mae, test_mae, best_parameterization
