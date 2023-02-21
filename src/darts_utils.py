from typing import List

import matplotlib.pyplot as plt
import numpy as np
from darts.metrics import mae, mape, mse, r2_score, rmse
from tqdm import tqdm
from utils import print_error_metrics


def backtest_local_models(
    models,
    scaled_series,
    past_covariates,
    future_covariates,
    forecast_horizon,
    start_split,
    verbose=False,
    retrain=False,
) -> List:
    return [
        model.historical_forecasts(
            series=serie,
            past_covariates=past_cov,
            future_covariates=future_cov,
            forecast_horizon=forecast_horizon,
            start=start_split,
            stride=1,
            retrain=retrain,
            last_points_only=True,
            overlap_end=False,
            verbose=verbose,
        )
        for model, serie, past_cov, future_cov in tqdm(
            list(zip(models, scaled_series, past_covariates, future_covariates))
        )
    ]


def backtest_global_model(
    model,
    scaled_series,
    past_covariates,
    future_covariates,
    forecast_horizon,
    start_split,
    verbose=False,
) -> List:
    backtests = [
        model.historical_forecasts(
            series=serie,
            past_covariates=past_cov,
            future_covariates=future_cov,
            forecast_horizon=forecast_horizon,
            start=start_split,
            stride=1,
            num_samples=100,
            retrain=False,
            last_points_only=True,
            overlap_end=False,
            verbose=verbose,
        )
        for serie, past_cov, future_cov in tqdm(
            list(zip(scaled_series, past_covariates, future_covariates))
        )
    ]
    return backtests


def fit_local_models(models, scaled_series, past_covariates, future_covariates) -> List:
    for model, serie, past_cov, future_cov in tqdm(
        list(zip(models, scaled_series, past_covariates, future_covariates))
    ):
        model.fit(
            series=serie[0],
            past_covariates=past_cov[0],
            future_covariates=future_cov[0],
        )
    return models


def full_fit_local_models(
    models, scaled_series, past_covariates, future_covariates
) -> List:
    for model, serie, past_cov, future_cov in tqdm(
        list(zip(models, scaled_series, past_covariates, future_covariates))
    ):
        model.fit(
            series=serie,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
    return models


def predict_local_models(
    models, forecast_horizon, past_covariates, future_covariates
) -> List:
    return [
        model.predict(
            n=forecast_horizon, past_covariates=past_cov, future_covariates=future_cov
        )
        for model, past_cov, future_cov in tqdm(
            list(zip(models, past_covariates, future_covariates))
        )
    ]


def predict_global_model(
    model, targets, forecast_horizon, past_covariates, future_covariates
) -> List:
    return [
        model.predict(
            n=forecast_horizon, past_covariates=past_cov, future_covariates=future_cov
        )
        for targets, past_cov, future_cov in tqdm(
            list(zip(targets, past_covariates, future_covariates))
        )
    ]


def fit_global_model(model, scaled_series, past_covariates, future_covariates) -> List:
    for serie, past_cov, future_cov in tqdm(
        list(zip(scaled_series, past_covariates, future_covariates))
    ):
        model.fit(
            series=serie[0],
            past_covariates=past_cov[0],
            future_covariates=future_cov[0],
        )
    return model


def calculate_loss(
    scalers, splited_series, backtests, log=False, scaling=False
) -> float:
    rmse_losses = list()
    mae_losses = list()
    for scaler, serie_list, backtest in tqdm(
        list(zip(scalers, splited_series, backtests))
    ):

        val_serie = serie_list[1]
        if scaling:
            val_serie = scaler.inverse_transform(val_serie)
            backtest = scaler.inverse_transform(backtest)

        if log:
            val_serie = val_serie.map(lambda x: (np.exp(x) - 1))
            backtest = backtest.map(lambda x: (np.exp(x) - 1))

        rmse_losses.append(rmse(val_serie.slice_intersect(backtest), backtest))
        mae_losses.append(mae(val_serie.slice_intersect(backtest), backtest))
    mean_rmse, std_rmse = np.mean(rmse_losses), np.std(rmse_losses)
    mean_mae, std_mae = np.mean(mae_losses), np.std(mae_losses)
    print(f"rmse_mean = {mean_rmse}, rmse_std = {std_rmse}")
    print(f"mae_mean = {mean_mae}, mae_std = {std_mae}")
    return mean_rmse


def inverse_forecasts(scalers, forecasts, log=False):
    scaled_forecasts = []
    for scaler, forecast in list(zip(scalers, forecasts)):
        forecast = scaler.inverse_transform(forecast)
        if log:
            forecast = forecast.map(lambda x: (np.exp(x) - 1))
        scaled_forecasts.append(forecast)
    return scaled_forecasts


def get_residuals(scalers, forecasts, series, log=False):
    residuals = []
    for scaler, forecast, serie in tqdm(list(zip(scalers, forecasts, series))):
        forecast = scaler.inverse_transform(forecast)
        serie = scaler.inverse_transform(serie)

        if log:
            forecast = forecast.map(lambda x: (np.exp(x) - 1))
            serie = serie.map(lambda x: (np.exp(x) - 1))

        residuals.append((forecast - serie.slice_intersect(forecast)).pd_dataframe())
    return residuals


def plot_prediction_forecasts(
    scalers, series, forecasts, slicing=True, log=False, scaling=False
) -> None:
    for scaler, serie, forecast in tqdm(list(zip(scalers, series, forecasts))):

        if scaling:
            serie = scaler.inverse_transform(serie)
            forecast = scaler.inverse_transform(forecast)

        if log:
            serie = serie.map(lambda x: (np.exp(x) - 1))
            forecast = forecast.map(lambda x: (np.exp(x) - 1))

        if slicing:
            serie.slice_intersect(forecast).plot(label="data")
        else:
            serie.plot(label="data")
        forecast.plot(lw=2, label="forecast")
        plt.legend()
        plt.show()


def plot_backtest_forecasts(
    scalers, splited_series, backtests, slicing=True, log=False, scaling=False
) -> None:
    for scaler, serie_list, backtest in tqdm(
        list(zip(scalers, splited_series, backtests))
    ):
        val_serie = serie_list[1]

        if scaling:
            val_serie = scaler.inverse_transform(val_serie)
            backtest = scaler.inverse_transform(backtest)

        if log:
            val_serie = val_serie.map(lambda x: (np.exp(x) - 1))
            backtest = backtest.map(lambda x: (np.exp(x) - 1))

        if slicing:
            val_serie.slice_intersect(backtest).plot(label="data")
        else:
            val_serie.plot(label="data")

        backtest.plot(lw=2, label="forecast")
        # covs.slice_intersect(backtest)[:slice_size].plot(label='covariates')
        # error = print_error_metrics(val_serie.slice_intersect(backtest).values(), backtest.values())

        # plt.title(f' MAE: {mae(val_serie,backtest)}, RMSE: {rmse(val_serie, backtest)}')
        # plt.title(error)
        plt.legend()
        plt.show()
