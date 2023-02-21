import numpy as np
import pandas as pd


# calculate different KPI
def upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["High"] - np.maximum(df["Close"], df["Open"])


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    return np.minimum(df["Close"], df["Open"]) - df["Low"]


def upper_shadow_percent(df: pd.DataFrame) -> pd.Series:
    return (df["High"] / np.maximum(df["Close"], df["Open"])) - 1


def lower_shadow_percent(df: pd.DataFrame) -> pd.Series:
    return (np.minimum(df["Close"], df["Open"]) / df["Low"]) - 1


def calculate_pct_returns(x: pd.Series, periods: int) -> pd.Series:
    return 1 + x.pct_change(periods=periods)


def calculate_cum_pct_returns(x: pd.Series, periods: int) -> pd.Series:
    return (((1 + x.pct_change(periods=periods)).cumprod() - 1)) * 100


def calculate_cum_log_returns(x: pd.Series, periods: int) -> pd.Series:
    return np.log(1 + x.pct_change(periods=periods)).cumsum()


def calculate_log_returns(x: pd.Series, periods: int) -> pd.Series:
    return np.log(1 + x.pct_change(periods=periods))
