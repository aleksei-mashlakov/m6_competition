import time
from datetime import datetime
import pandas as pd
import numpy as np

# adapted from https://learndataanalysis.org/source-code-download-historical-stock-data-from-yahoo-finance-using-python/
def get_ticker_historical_data(ticker: str, 
                               from_date: datetime,
                               to_date: datetime,
                               interval: str = '1d'
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
    interval = '1d' # 1wk, 1mo
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={from_date}&period2={to_date}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string, index_col=0, parse_dates=[0], infer_datetime_format=True)
    return df


def calculate_mape(y_true, y_pred):
    """ Calculate mean absolute percentage error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_mpe(y_true, y_pred):
    """ Calculate mean percentage error (MPE)"""
    return np.mean((y_true - y_pred) / y_true) * 100

def calculate_mae(y_true, y_pred):
    """ Calculate mean absolute error (MAE)"""
    return np.mean(np.abs(y_true - y_pred)) #* 100

def calculate_rmse(y_true, y_pred):
    """ Calculate root mean square error (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred)**2)) #/ np.sum(y_true)

def print_error_metrics(y_true, y_pred):
    print('MAPE: %f'%calculate_mape(y_true, y_pred))
    print('MPE: %f'%calculate_mpe(y_true, y_pred))
    print('MAE: %f'%calculate_mae(y_true, y_pred))
    print('RMSE: %f'%calculate_rmse(y_true, y_pred))
    
