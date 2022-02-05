from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
# from typing import boolean        
# adapted from https://github.com/mjmacarty/alphavantage/blob/main/3-momentum_algorithmic.ipynb


class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")
    
@dataclass
class SMAStrategy(Strategy):
    prefix = 'sma'
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        If the shorter-term SMA value is greater than the longer-term one ... go long on the stock (put a 1).
        Otherwise, go short on the stock (put a -1).
        """
        data['ma_short'] = data[self.close_name].rolling(window=self.short_window).mean().shift()
        data['ma_long'] = data[self.close_name].rolling(window=self.long_window).mean().shift()
        data['ma_signal'] = np.where(data['ma_short'] > data['ma_long'], 1, 0)
        data['ma_signal'] = np.where(data['ma_short'] < data['ma_long'], -1, data['ma_signal'])
        data.dropna(inplace=True)
        return data
    
@dataclass
class EMAStrategy(Strategy):
    prefix = 'ema'
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        """
        data['ma_short'] = data[self.close_name].ewm(span=self.short_window, adjust=False).mean().shift()
        data['ma_long'] = data[self.close_name].ewm(span=self.long_window, adjust=False).mean().shift()
        data['ma_signal'] = np.where(data['ma_short'] > data['ma_long'], 1, 0)
        data['ma_signal'] = np.where(data['ma_short'] < data['ma_long'], -1, data['ma_signal'])
        data.dropna(inplace=True)
        return data
    
    
@dataclass
class TestStrategy(object):
    
    strategy: Strategy
    short_window: int = 50
    long_window: int = 200
    close_name: str = 'Close'
    name: str = 'sma'
        
        
    def __post_init__(self):
        self.strategy.short_window=self.short_window
        self.strategy.long_window=self.long_window
        self.strategy.close_name=self.close_name
    
        
    def run(self, data: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
        data = self.strategy.get_signals(data=data)
        data = self.get_returns(data=data)
        if plot:
            self.plot_strategy(data=data)
            self.plot_returns(data=data)
        return data
    
    def get_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        data['ma_return'] = np.log(1 + data[self.close_name].pct_change())    
        data['ma_system_return'] = data['ma_signal'] * data['ma_return']
        data['ma_entry'] = data['ma_signal'].diff()
        print(f"Buy/sell returns: {np.exp(data['ma_return'].cumsum())[-1]-1}")
        print(f"System returns: {np.exp(data['ma_system_return'].cumsum())[-1] -1}")
        return data
    
    def get_returns2(self, data: pd.DataFrame) -> pd.DataFrame:
        data['ma_return'] = np.log(data[self.close_name]).diff()    
        data['ma_system_return'] = data['ma_signal'] * data['ma_return']
        data['ma_entry'] = data['ma_signal'].diff()
        print(f"Buy/sell returns: {np.exp(data['ma_return']).cumprod()[-1]-1}")
        print(f"System returns: {np.exp(data['ma_system_return']).cumprod()[-1] -1}")
        return data
    
    def plot_strategy(self, data: pd.DataFrame):
        plt.rcParams['figure.figsize'] = 12, 6
        plt.grid(True, alpha = .3)
        plt.plot(data.iloc[-252:]['Adj Close'], label = 'GLD')
        plt.plot(data.iloc[-252:]['ma_short'], label = 'ma_short')
        plt.plot(data.iloc[-252:]['ma_long'], label = 'ma_long')
        plt.plot(data[-252:].loc[data['ma_entry'] == 2].index, data[-252:]['ma_short'][data['ma_entry'] == 2], '^',
                 color = 'g', markersize = 12)
        plt.plot(data[-252:].loc[data['ma_entry'] == -2].index, data[-252:]['ma_long'][data['ma_entry'] == -2], 'v',
                 color = 'r', markersize = 12)
        plt.legend(loc=2);
        plt.show()
        return 
    
    def plot_returns(self, data: pd.DataFrame) -> None:
        plt.plot(np.exp(data['ma_return']).cumprod()-1, label='Buy/Hold')
        plt.plot(np.exp(data['ma_system_return']).cumprod()-1, label='System')
        plt.legend(loc=2)
        plt.grid(True, alpha=.3)
        plt.show()
        return 
