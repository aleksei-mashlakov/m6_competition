from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from pytorch_forecasting.metrics import MultiHorizonMetric


def calculate_mape(y_true, y_pred):
    """Calculate mean absolute percentage error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_mpe(y_true, y_pred):
    """Calculate mean percentage error (MPE)"""
    return np.mean((y_true - y_pred) / y_true) * 100


def calculate_mae(y_true, y_pred):
    """Calculate mean absolute error (MAE)"""
    return np.mean(np.abs(y_true - y_pred))  # * 100


def calculate_rmse(y_true, y_pred):
    """Calculate root mean square error (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))  # / np.sum(y_true)


def print_error_metrics(y_true, y_pred):
    print("MAPE: %f" % calculate_mape(y_true, y_pred))
    print("MPE: %f" % calculate_mpe(y_true, y_pred))
    print("MAE: %f" % calculate_mae(y_true, y_pred))
    print("RMSE: %f" % calculate_rmse(y_true, y_pred))


def calculate_rps(probs: npt.ArrayLike, outcome: npt.ArrayLike) -> float:
    """
    Outcome and Probs must be provided with the same order as probabilities.

    Args:
        probs (npt.ArrayLike): Probs should be a list of probabilities. [0.79, 0.09, 0.12] for example.
        outcome (npt.ArrayLike): Outcome should be a binary list of the ordinal outcome. [0, 1, 0] for exmaple.

    Returns:
        float: rank probability score
    """

    return np.mean(((np.cumsum(probs) - np.cumsum(outcome)) ** 2), axis=0)


def portfolio_rps(probs: npt.NDArray, outcome: npt.NDArray) -> float:
    all_rps = np.array(
        [calculate_rps(probs[i, :], outcome[i, :]) for i in range(outcome.shape[0])]
    )
    return np.mean(all_rps, axis=0)


def torch_rps(probs: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    return torch.mean(
        ((torch.cumsum(probs, dim=-1) - torch.cumsum(outcome, dim=-1)) ** 2),
        dim=-1,
        keepdim=True,
    )


class RPS(MultiHorizonMetric):
    def loss(self, y_pred, target):
        y_pred = F.softmax(y_pred, dim=-1)
        target = F.one_hot(target)
        loss = torch_rps(y_pred, target)
        return loss

    def to_quantiles(self, out: Dict[str, torch.Tensor], quantiles=None):
        return out

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        return y_pred


###########################################################################
#  Code for computing the RPS and IR scores for a given evaluation period
###########################################################################

# For simplicity, in this example it is assumed that the data provided cover a single evaluation period.
# This period is specified through the min/max date of the asset prices data set.
# If you wish to compute RPS/IR for multiple periods, you'll have to execute
# the script multiple times, each time using a different, appropriate input.

import glob
from statistics import stdev

import numpy as np
import pandas as pd


# https://www.kaggle.com/code/marcogorelli/m6-calculation-example-70x-faster?scriptVersionId=106287617
# Function for computing RPS
def RPS_calculation(hist_data, submission, asset_no=100):

    if hist_data.shape[0] <= asset_no:
        return np.nan

    asset_id = pd.unique(hist_data.symbol)

    # Compute percentage returns
    asset_id = sorted(asset_id)

    returns = (
        hist_data.groupby("symbol")["price"]
        .agg(("last", "first"))
        .pipe(lambda df: (df["last"] - df["first"]) / df["first"])
        .rename_axis(index="ID")
        .to_frame(name="Return")
        .reset_index()
    )

    # Define the relevant position of each asset
    ranking = pd.DataFrame(columns=["ID", "Position", "Return"])
    ranking.ID = list(asset_id)
    ranking.Return = returns.Return
    ranking.Position = ranking.Return.rank(method="min")

    # Handle Ties
    Series_per_position = pd.DataFrame(
        columns=[
            "Position",
            "Series",
            "Rank",
            "Rank1",
            "Rank2",
            "Rank3",
            "Rank4",
            "Rank5",
        ]
    )
    Series_per_position.Position = list(
        pd.unique(ranking.Position.sort_values(ascending=True))
    )

    if len(Series_per_position) == len(asset_id):
        # no ties, use fast code
        returns["Rank"] = (
            pd.qcut(returns["Return"], q=[0, 0.2, 0.4, 0.6, 0.8, 1]).cat.codes + 1
        )
        ranking = pd.concat(
            [
                ranking,
                returns["Rank"],
                pd.get_dummies(returns["Rank"], prefix="Rank")
                .rename(columns=lambda x: x.replace("_", ""))
                .astype(float),
            ],
            axis=1,
        )
    else:
        ranking = slow_rps_calculation(ranking, submission)

    rps_sub = (
        (
            (
                submission.set_index("ID").filter(regex=r"Rank\d").cumsum(axis=1)
                - ranking.set_index("ID").filter(regex=r"Rank\d").cumsum(axis=1)
            )
            ** 2
        )
        .mean(axis=1)
        .mean()
    )

    output = {"RPS": rps_sub, "details": submission}

    return output, ranking


# Function for computing IR
def IR_calculation(hist_data, submission):

    asset_id = pd.unique(hist_data.symbol)
    asset_id = sorted(asset_id)

    # Compute percentage returns
    # returns = pd.DataFrame(columns=["ID", "Return"])

    # Investment weights
    weights = submission[["ID", "Decision"]]

    pivoted = (
        hist_data.pivot_table(index=["date"], columns=["symbol"], values=["price"])
        .pct_change(axis=0)
        .iloc[1:]
    )
    pivoted.columns = pivoted.columns.get_level_values(1)
    RET = np.log(
        1
        + (
            pivoted.reindex(columns=asset_id)
            * weights.set_index("ID").reindex(asset_id)["Decision"].to_numpy()
        ).sum(axis=1)
    ).reset_index(drop=True)
    return RET


def ffill_missing_prices(hist_data):
    pivoted = hist_data.pivot_table(
        index=["date"], columns=["symbol"], values=["price"]
    ).fillna(method="ffill", axis=0)
    pivoted.columns = pivoted.columns.get_level_values(1)
    pivoted = pivoted.reset_index()
    hist_data = pivoted.melt(id_vars=["date"], value_name="price")
    return hist_data


# Read asset prices data (as provided by the M6 submission platform)
# asset_data = pd.read_csv(glob.glob('../input/m6naivesubmissionfiles/assets_m6*.csv')[0])
# asset_data['date'] = pd.to_datetime(asset_data['date'])
# asset_data = ffill_missing_prices(asset_data)
# run_evaluation([1, 2, 3], asset_data)
# from https://www.kaggle.com/code/marcogorelli/m6-calculation-example-70x-faster?scriptVersionId=102662740


def run_evaluation(periods, asset_data):
    rpss = []
    rets = []
    for i in periods:
        start = pd.Timestamp("2022-03-04") + pd.Timedelta(days=28 * (i - 1))
        end = start + pd.Timedelta(days=28)
        print(start, end)
        asset_data_ = asset_data[asset_data["date"].between(start, end)].copy()

        try:
            submission_data = pd.read_csv(
                f"../input/m6naivesubmissionfiles/submission_{i}.csv"
            )
        except FileNotFoundError:
            # just use default submission
            submission_data = pd.read_csv(
                f"../input/m6naivesubmissionfiles/submission_1.csv"
            )
        submission_data["ID"] = submission_data["ID"].replace("FB", "META")
        rpss.append(
            RPS_calculation(hist_data=asset_data_, submission=submission_data)["RPS"]
        )
        rets.append(IR_calculation(asset_data_, submission_data))
    print(f"RPS: {np.mean(rpss)}")
    print(f"IR (numerator): {pd.concat(rets).sum()}")
    print(f"IR (denominator): {pd.concat(rets).std()}")
    print(f"IR: {pd.concat(rets).sum() / pd.concat(rets).std()}")


def slow_rps_calculation(ranking, submission):
    print("Using slow RPS calculation")
    temp = ranking.Position.value_counts()
    temp = pd.DataFrame(zip(temp.index, temp), columns=["Rank", "Occurencies"])
    temp = temp.sort_values(by=["Rank"], ascending=True)
    Series_per_position.Series = list(temp.Occurencies)
    Series_per_position

    total_ranks = Series_per_position.Position.values[-1]
    for i in range(0, Series_per_position.shape[0]):

        start_p = Series_per_position.Position[i]
        end_p = Series_per_position.Position[i] + Series_per_position.Series[i]
        temp = pd.DataFrame(
            columns=["Position", "Rank", "Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]
        )
        temp.Position = list(range(int(start_p), int(end_p)))

        if (
            temp.loc[
                temp.Position.isin(list(range(1, int(0.2 * total_ranks + 1))))
            ].empty
            == False
        ):
            temp.loc[
                temp.Position.isin(list(range(1, int(0.2 * total_ranks + 1))))
            ] = temp.loc[
                temp.Position.isin(list(range(1, int(0.2 * total_ranks + 1))))
            ].assign(
                Rank=1
            )
            temp.loc[
                temp.Position.isin(list(range(1, int(0.2 * total_ranks + 1))))
            ] = temp.loc[
                temp.Position.isin(list(range(1, int(0.2 * total_ranks + 1))))
            ].assign(
                Rank1=1.0
            )

        elif (
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.2 * total_ranks + 1), int(0.4 * total_ranks + 1)))
                )
            ].empty
            == False
        ):
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.2 * total_ranks + 1), int(0.4 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.2 * total_ranks + 1), int(0.4 * total_ranks + 1)))
                )
            ].assign(
                Rank=2
            )
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.2 * total_ranks + 1), int(0.4 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.2 * total_ranks + 1), int(0.4 * total_ranks + 1)))
                )
            ].assign(
                Rank2=1.0
            )

        elif (
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.4 * total_ranks + 1), int(0.6 * total_ranks + 1)))
                )
            ].empty
            == False
        ):
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.4 * total_ranks + 1), int(0.6 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.4 * total_ranks + 1), int(0.6 * total_ranks + 1)))
                )
            ].assign(
                Rank=3
            )
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.4 * total_ranks + 1), int(0.6 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.4 * total_ranks + 1), int(0.6 * total_ranks + 1)))
                )
            ].assign(
                Rank3=1.0
            )

        elif (
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.6 * total_ranks + 1), int(0.8 * total_ranks + 1)))
                )
            ].empty
            == False
        ):
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.6 * total_ranks + 1), int(0.8 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.6 * total_ranks + 1), int(0.8 * total_ranks + 1)))
                )
            ].assign(
                Rank=4
            )
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.6 * total_ranks + 1), int(0.8 * total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.6 * total_ranks + 1), int(0.8 * total_ranks + 1)))
                )
            ].assign(
                Rank4=1.0
            )

        elif (
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.8 * total_ranks + 1), int(total_ranks + 1)))
                )
            ].empty
            == False
        ):
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.8 * total_ranks + 1), int(total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.8 * total_ranks + 1), int(total_ranks + 1)))
                )
            ].assign(
                Rank=5
            )
            temp.loc[
                temp.Position.isin(
                    list(range(int(0.8 * total_ranks + 1), int(total_ranks + 1)))
                )
            ] = temp.loc[
                temp.Position.isin(
                    list(range(int(0.8 * total_ranks + 1), int(total_ranks + 1)))
                )
            ].assign(
                Rank5=1.0
            )
        temp = temp.fillna(0)
        Series_per_position.iloc[i, 2 : Series_per_position.shape[1]] = temp.mean(
            axis=0
        ).iloc[1 : temp.shape[1]]

    Series_per_position = Series_per_position.drop("Series", axis=1)
    ranking = pd.merge(ranking, Series_per_position, on="Position")
    ranking = ranking[
        [
            "ID",
            "Return",
            "Position",
            "Rank",
            "Rank1",
            "Rank2",
            "Rank3",
            "Rank4",
            "Rank5",
        ]
    ]
    ranking = ranking.sort_values(["Position"])
    return ranking
