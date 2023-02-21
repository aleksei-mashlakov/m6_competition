from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import optuna as opt
import pandas as pd
import riskfolio as rp
from riskfolio.ParamsEstimation import covar_matrix, mean_vector

from src.metrics import IR_calculation, RPS_calculation

ir = lambda rets: pd.concat(rets).sum() / pd.concat(rets).std()


@dataclass
class PortfolioConfig:
    sht: bool  # Allows to use Short Weights
    uppersht: float  # Maximum value of sum of short weights in absolute value
    upperlng: float  # Maximum value of sum of positive weights
    alpha: float  # Significance level for CVaR, EVaR y CDaR
    budget: Optional[float] = None

    def __post_init__(self):
        self.budget = self.upperlng - self.uppersht

    def get_risk_portfolio(self) -> rp.Portfolio:
        port = rp.Portfolio()
        port.sht = self.sht  # Allows to use Short Weights
        port.uppersht = (
            self.uppersht
        )  # Maximum value of sum of short weights in absolute value
        port.upperlng = self.upperlng  # Maximum value of sum of positive weights
        port.budget = port.upperlng - port.uppersht
        port.alpha = self.alpha  # Significance level for CVaR, EVaR y CDaR
        return port


@dataclass
class PortfolioOptConfig:
    cov: str
    mu: str
    model: str
    obj: str
    rm: str
    weeks_lookback: int
    kelly: Union[str, bool]


def backtest_M6_ir(
    port_params: PortfolioConfig,
    opt_config: PortfolioOptConfig,
    m6_price_data,
    returns_data,
    df_submission,
    start: str = "2022-03-04",
    end: datetime = pd.Timestamp.today(),
):
    column = opt_config.mu + "_" + opt_config.cov + "_" + opt_config.rm
    print(column)
    returns = []

    # Building the portfolio object
    port = port_params.get_risk_portfolio()

    for start in pd.date_range(start=start, end=end, freq="28D"):
        # start = pd.Timestamp("2022-03-04") + pd.Timedelta(days=28 * (i - 1))
        end = start + pd.Timedelta(days=28)
        asset_data_ = m6_price_data[m6_price_data["date"].between(start, end)].copy()
        asset_data_fit = returns_data[
            (
                returns_data.index
                > (start - pd.Timedelta(days=7 * opt_config.weeks_lookback))
            )
            & (returns_data.index < start)
        ].copy()
        port.returns = asset_data_fit.dropna()
        port.assets_stats(method_mu=opt_config.mu, method_cov=opt_config.cov, d=0.95)
        w = port.optimization(
            model=opt_config.model,
            rm=opt_config.rm,
            obj=opt_config.obj,
            rf=0,
            l=0,
            hist=False,
            # kelly=opt_config.kelly,
        )
        if w is None:
            print("WARNING: weights are None")
            return -10
        # print(f"Weights abs sum: {w.abs().sum()[0]}")
        w_sum = w.abs().sum()[0]
        if w_sum < 0.75:
            print(f"WARNING: weights' abs sum is below 75%: {round(w_sum,2)*100}%")

        ## Get returns
        w = w.reindex(index=df_submission.ID.values)
        submission_data = df_submission.copy()
        submission_data["Decision"] = w.values
        returns.append(IR_calculation(asset_data_, submission_data))

    quarter_ir = [ir(returns[i : i + 3]) for i in range(0, 12, 3) if i < len(returns)]
    ir_sum = ir(returns)
    for i in range(len(quarter_ir)):
        print(f"{column} Q{i+1}: IR: {quarter_ir[i]}")
    print(f"{column} Global IR: {ir_sum}")
    return ir_sum


def scores_to_quintiles(x):
    ys = list()
    q = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    for xi in x:
        y = np.searchsorted(q, xi)
        ys.append(y)
    return np.array(ys)


def mvn_quintile_probabilities(sigma, n_samples, mu=None):
    n_dim = np.shape(sigma)[0]
    if mu is None:
        mu = np.zeros(n_dim)
    x = np.random.multivariate_normal(
        mu, sigma, size=n_samples, check_valid="warn", tol=1e-8
    )
    y = scores_to_quintiles(x)
    p = list()
    for i in range(5):
        pi = np.mean(y == i, axis=0)
        p.append(pi)
    return p


def backtest_M6_rps(
    opt_config: PortfolioOptConfig,
    n_samples: int,
    assets: List[str],
    m6_price_data,
    returns_data,
    df_submission,
) -> float:
    column = opt_config.mu + "_" + opt_config.cov
    rpss = []
    for start in pd.date_range(
        start="2022-03-04", end=pd.Timestamp.today(), freq="28D"
    ):
        # start = pd.Timestamp("2022-03-04") + pd.Timedelta(days=28 * (i - 1))
        end = start + pd.Timedelta(days=28)
        asset_data_ = m6_price_data[m6_price_data["date"].between(start, end)].copy()
        asset_data_fit = returns_data[
            (
                returns_data.index
                > (start - pd.Timedelta(days=7 * opt_config.weeks_lookback))
            )
            & (returns_data.index < start)
        ].copy()
        # mu = mean_vector(X=asset_data_fit, method=opt_config.mu).to_numpy().reshape(-1)
        cov = covar_matrix(X=asset_data_fit, method=opt_config.cov)
        mu = np.diag(cov).reshape(-1)
        # print(cov)
        probs = mvn_quintile_probabilities(
            sigma=cov.to_numpy(), mu=mu, n_samples=n_samples
        )
        rsps = pd.DataFrame(columns=df_submission.ID.values, data=probs).transpose()
        submission_data = df_submission.copy()
        idx = submission_data.loc[submission_data["ID"].isin(assets)].index.values
        submission_data.iloc[idx, 1:-1] = rsps.loc[assets, :].values
        rsp = RPS_calculation(asset_data_, submission_data)["RPS"]
        print(f"Month {i} RSP={rsp}")
        rpss.append(rsp)
    mean_rps = np.mean(rpss)
    print(f"{column} RPS: {mean_rps}")
    return mean_rps


def backtest_M6_rps_isotonic(
    opt_config: PortfolioOptConfig,
    n_samples: int,
    m6_price_data,
    returns_data,
    df_submission,
) -> float:
    column = opt_config.mu + "_" + opt_config.cov
    rpss = []
    for i in range(1, 8):
        start = pd.Timestamp("2022-03-04") + pd.Timedelta(days=28 * (i - 1))
        end = start + pd.Timedelta(days=28)
        asset_data_ = m6_price_data[m6_price_data["date"].between(start, end)].copy()
        asset_data_fit = returns_data[
            (
                returns_data.index
                > (start - pd.Timedelta(days=7 * opt_config.weeks_lookback))
            )
            & (returns_data.index < start)
        ].copy()

        probs = (asset_data_fit.apply(pd.value_counts) / len(asset_data_fit)).T
        # rsps = pd.DataFrame(columns=df_submission.ID.values, data=probs).transpose()
        submission_data = df_submission.copy()
        submission_data.iloc[:, 1:-1] = probs.values
        rsp = RPS_calculation(asset_data_, submission_data)["RPS"]
        print(f"Month {i} RSP={rsp}")
        rpss.append(rsp)
    mean_rps = np.mean(rpss)
    print(f"{column} RPS: {mean_rps}")
    return mean_rps


def logging_callback(study: opt.Study, frozen_trial: opt.Trial):
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
    else:
        print(
            "Trial {} finished with value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
            )
        )
