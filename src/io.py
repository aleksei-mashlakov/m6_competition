import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import pandas_datareader as pdr
from tqdm.notebook import tqdm


def get_m6_tickers_data(
    tickers: List[str],
    from_date: datetime = pd.to_datetime("2018-01-01"),
    to_date: datetime = pd.Timestamp.today(),
    download_dir: str = "./data/tickers",
    interval: str = "1d",
    save_data: bool = False,
) -> Dict[str, pd.DataFrame]:

    to_date.tz_localize(tz="Europe/Moscow").tz_convert(tz="America/New_York")
    to_date.replace(hour=0, minute=0, second=0, microsecond=0)

    tickers_data = dict()
    for ticker in tqdm(tickers[:]):
        #     data = get_ticker_historical_data(ticker=ticker,
        #                                       from_date=from_date,
        #                                       to_date=to_date,
        #                                       interval=interval
        #                                       )
        # This returns a data frame of scraped stock data from yahoo
        data = pdr.DataReader(ticker, "yahoo", from_date, to_date)
        tickers_data[ticker] = data

    if save_data:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ticker, data in tickers_data:
            data.reset_index().to_csv(
                os.path.join(download_dir, f"{ticker}_{interval}.csv")
            )
    return tickers_data


# adapted from https://learndataanalysis.org/source-code-download-historical-stock-data-from-yahoo-finance-using-python/
def get_ticker_historical_data(
    ticker: str, from_date: datetime, to_date: datetime, interval: str = "1d"
) -> pd.DataFrame:
    """
    Returns a dataframe of historical values

    Example use:

    >>> df = get_ticker_historical_data(ticker='AAPL',
                                        from_date=pd.to_datetime("2021-12-1"),
                                        to_date=pd.to_datetime("2021-12-31"),
                                        interval='1d'
                                        )
    >>> df.head()
    """
    from_date = int(time.mktime(from_date.timetuple()))
    to_date = int(time.mktime(to_date.timetuple()))
    interval = "1d"  # 1wk, 1mo
    query_string = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={from_date}&period2={to_date}&interval={interval}&events=history&includeAdjustedClose=true"
    df = pd.read_csv(
        query_string, index_col=0, parse_dates=[0], infer_datetime_format=True
    )
    return df


def get_dre_ticker_data():
    """Get DRE ticker data from text file."""
    # Free stock data was available from https://stooq.com/db/h/
    # read text file into pandas DataFrame
    df = pd.read_csv(
        "./data/raw/dre.us.txt"
    )  # , sep="\t", header=0, parse_dates=[0], index_col=0)
    df.drop(["<TICKER>", "<PER>", "<TIME>", "<OPENINT>"], axis=1, inplace=True)
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Adj Close"] = df["Close"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.set_index("Date", inplace=True)
    df = df.loc[:, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    # read text file into pandas DataFrame
    df_dre_missing = pd.read_csv(
        "./data/raw/dre.csv", sep="\t", header=0, parse_dates=[0], index_col=0
    )
    df_dre_missing.drop(["Change", "Change (%)"], axis=1, inplace=True)
    df_dre_missing["Adj Close"] = df_dre_missing["Close"]
    df_dre_missing["Volume"] = df_dre_missing["Volume"].str.replace(",", "").astype(int)
    df_dre_missing = df_dre_missing.loc[
        df_dre_missing.index[::-1],
        ["Open", "High", "Low", "Close", "Adj Close", "Volume"],
    ]

    dre_ticker_data = pd.concat([df, df_dre_missing])
    dre_ticker_data[["Open", "High", "Low", "Close"]] = dre_ticker_data[
        ["Open", "High", "Low", "Close"]
    ].astype(float)
    dre_ticker_data["Volume"] = dre_ticker_data["Volume"].astype(int)
    return dre_ticker_data


def get_today_date():
    return (
        pd.Timestamp.today()
        .tz_localize(tz="Europe/Moscow")
        .tz_convert(tz="America/New_York")
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )
