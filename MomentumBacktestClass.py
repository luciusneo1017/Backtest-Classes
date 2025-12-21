import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
import os

class MomentumVectorBacktest():

    def __init__(self,symbol:str,momentum:int,start:str,end:str,tc:float):
        '''
        Class for vectorised backtesting of a Momentum based trading strategy
    
        '''
        self.symbol = symbol
        self.tc = tc # transaction costs (decimal, e.g. 0.001 = 0.1%; values >1 treated as percent, tc = 0.0 models no transaction costs scenario)
        self.results = None
        self.start = start
        self.end = end
        self.momentum = momentum
        if self.momentum <= 0:
            raise ValueError("momentum must be a positive integer")
        self.data = self.get_data()

    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end, auto_adjust = True)
        if isinstance(raw.columns, pd.MultiIndex):
            price = raw['Close'][self.symbol]
        else:
            price = raw['Close']
        raw = price.rename('price').to_frame()
        return raw
    
    def run_strategy(self):
        '''
        Backtests the trading strategy
        '''
        momentum = self.momentum
        data = self.data.copy().dropna()
        data['return'] = np.log(data['price']/data['price'].shift(1))
        data['signal'] = np.sign(data['return'].rolling(momentum).sum())
        data['position'] = data['signal'].shift(1) # executed position (lagged to avoid look-ahead)
        data.dropna(inplace = True)
        data['strategy'] = data['position'] * data['return']

        turnover = data['position'].diff().abs().fillna(0.0) # number of trades (2 when flipping sign)
        tc = self.tc / 100 if self.tc > 1 else float(self.tc)
        if tc:
            if not (0 <= tc < 1):
                raise ValueError("Transaction cost must be between 0 and 1 (or 0 and 100 if in %).")
            data["strategy"] += turnover * np.log(1 - tc)

        data['creturns'] =  np.exp(data['return'].cumsum())
        data['cstrategy'] = np.exp(data['strategy'].cumsum())
        self.results = data

        bh_factor = data['creturns'].iloc[-1]       # gross performance of asset
        strat_factor = data['cstrategy'].iloc[-1]   # gross performance of strategy
        alpha_ret = (strat_factor - 1) - (bh_factor - 1) # out/under-performance of strategy
        
        return (f"buy&hold: {bh_factor - 1:.2%}, "
                f"strategy: {strat_factor - 1:.2%}, "
                f"alpha: {alpha_ret:.2%}") #alpha is defined by out/under-performance of strategy as opposed to the gross return of asset over the period in this case
    
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        title = f"{self.symbol} | Momentum = {self.momentum} | TC = {self.tc} "
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = (10,6))
        plt.show()

    def max_drawdown(self): # Implementing Max Drawdown with Sliding Window
        if self.results is None:
            raise ValueError("Run the strategy first (run_strategy).")

        equity = self.results["cstrategy"].to_numpy()
        peak = equity[0]
        max_dd = 0.0

        for x in equity[1:]:
            peak = max(peak, x)
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)

        print(f"Max Drawdown: {max_dd:.2%}")
        return max_dd


