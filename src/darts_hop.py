import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR, NuSVR


class Objective(object):
    def __init__(
        self,
        scaled_series,
        past_covariates,
        future_covariates,
        splited_series,
        splited_past_covariates,
        splited_future_covariates,
        forecast_horizon,
        scalers,
    ):
        # Hold this implementation specific arguments as the fields of the class.
        self.scaled_series = scaled_series
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates
        self.splited_series = splited_series
        self.splited_past_covariates = splited_past_covariates
        self.splited_future_covariates = splited_future_covariates
        self.forecast_horizon = forecast_horizon
        self.scalers = scalers

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.

        regressor_type = trial.suggest_categorical(
            "regressor", ["SVC", "LinearSVR", "NuSVR"]
        )
        # gammas = trial.suggest_categorical("regressor", ['scale', 'auto'])
        # kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        svc_c = trial.suggest_float("svc_c", 1e-3, 1e2, log=True)
        svc_epsilon = trial.suggest_float("svc_epsilon", 1e-1, 1e1, log=True)

        if regressor_type == "SVC":
            model = SVR(
                kernel="poly",
                C=svc_c,
                gamma="auto",
                degree=3,
                epsilon=svc_epsilon,
                coef0=1,
                random_state=nrd,
            )
            # svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
            # svr_lin = SVR(kernel="linear", C=100, gamma="auto")
            # model = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

        elif regressor_type == "LinearSVR":
            model = LinearSVR(
                C=svc_c,
                gamma="auto",
                degree=3,
                epsilon=svc_epsilon,
                coef0=1,
                random_state=nrd,
            )

        elif regressor_type == "NuSVR":
            svr_nu = trial.suggest_float("svr_nu", 1e-1, 1e0, log=True)
            model = NuSVR(
                nu=svr_nu,
                kernel="poly",
                C=svc_c,
                gamma="auto",
                degree=3,
                epsilon=svc_epsilon,
                coef0=1,
                random_state=nrd,
            )

        models = fit_local_models(
            models,
            self.splited_series,
            self.splited_past_covariates,
            self.splited_future_covariates,
        )
        backtests = backtest_local_models(
            models,
            self.scaled_series,
            self.past_covariates,
            self.future_covariates,
            self.forecast_horizon,
            start_split=0.9,
        )

        loss = calculate_loss(self.scalers, self.splited_series, backtests)

        return loss


import optuna
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (  # TweedieRegressor,; SGDRegressor,
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    RidgeCV,
)

# SEED = 42
# nrd = np.random.seed(SEED)

# # Turn off optuna log notes.
# optuna.logging.set_verbosity(optuna.logging.WARN)


def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )


class Objective(object):
    def __init__(
        self,
        scaled_series,
        past_covariates,
        future_covariates,
        splited_series,
        splited_past_covariates,
        splited_future_covariates,
        forecast_horizon,
        scalers,
        lags_future_1=7,
        lags_future_2=2,
        lags=14,
        lags_past=14,
    ):
        # Hold this implementation specific arguments as the fields of the class.
        self.scaled_series = scaled_series
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates
        self.splited_series = splited_series
        self.splited_past_covariates = splited_past_covariates
        self.splited_future_covariates = splited_future_covariates
        self.forecast_horizon = forecast_horizon
        self.scalers = scalers
        self.lags_future_1 = lags_future_1
        self.lags_future_2 = lags_future_2
        self.lags = lags
        self.lags_past = lags_past

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.

        # alpha = trial.suggest_float("alpha", 1e-3, 1e0, log=True)
        # l1_ratio = trial.suggest_categorical("l1_ratio", [.1, .25, .5, .7, .9, .95, .99, 1])
        l1_ratio = trial.suggest_float("l1_ratio", 1e-1, 1e0, log=True)
        # lags_past = trial.suggest_int("lags_past", 2, 14, log=True)
        # lags = trial.suggest_int("lags", 2, 14, log=True)
        # lags_future_1 = trial.suggest_int("lags_future_1", 1, 7, log=True)
        # lags_future_2 = trial.suggest_int("lags_future_2", 0, 5, log=False, step=1)

        # init the models
        models = [
            RegressionModel(
                lags=self.lags,
                lags_past_covariates=self.lags_past,
                lags_future_covariates=(self.lags_future_1, self.lags_future_2),
                model=ElasticNetCV(  # alphas=alphas,
                    l1_ratio=l1_ratio, random_state=nrd
                ),
            )
            for model in range(len(self.scaled_series))
        ]

        models = fit_local_models(
            models,
            self.splited_series,
            self.splited_past_covariates,
            self.splited_future_covariates,
        )
        backtests = backtest_local_models(
            models,
            self.scaled_series,
            self.past_covariates,
            self.future_covariates,
            self.forecast_horizon,
            start_split=0.9,
        )

        loss = calculate_loss(self.scalers, self.splited_series, backtests)
        return loss


# study = optuna.create_study(
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(seed=SEED),
#     # storage="sqlite:///example.db"
# )
# objective = Objective(
#     scaled_series=scaled_series,
#     past_covariates=past_covariates,
#     future_covariates=future_covariates,
#     splited_series=splited_series,
#     splited_past_covariates=splited_past_covariates,
#     splited_future_covariates=splited_future_covariates,
#     forecast_horizon=forecast_horizon,
#     scalers=scalers,
# )
# study.optimize(objective, n_trials=100, callbacks=[logging_callback])

# print(f"Best trial: \n{study.best_trial}\n")
# print(f"Best value: {study.best_value}\n")
# print(f"Best params: {study.best_params}\n")
# # print(study.trials)

# study_name = "ElasticNetCV"
# results_directory = "./results"
# if not os.path.exists(results_directory):
#     os.makedirs(results_directory)
# study.trials_dataframe().to_csv(f"{results_directory}/{study_name}.csv", index=False)
# plot_parallel_coordinate(study).write_html(
#     f"{results_directory}/{study_name}_parallel.html"
# )
# plot_contour(study).write_html(f"{results_directory}/{study_name}_contour.html")

# from optuna.visualization import (plot_contour, plot_edf,
#                                   plot_intermediate_values,
#                                   plot_optimization_history,
#                                   plot_parallel_coordinate,
#                                   plot_param_importances, plot_slice)

# plot_contour(study)
# plot_intermediate_values(study)
# plot_parallel_coordinate(study)
