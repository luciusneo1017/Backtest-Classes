import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
import os

class MomentumVectorBacktest():

    def __init__(self,symbol,start,end,amount,tc,pull_data):
        '''
        Class for vectorised backtesting of a Momentum based trading strategy
    
        '''
        self.symbol = symbol
        self.tc = tc # transaction costs (decimal, e.g. 0.001 = 0.1%; values >1 treated as percent)
        self.results = None
        self.amount = amount # amount to be invested
        if pull_data:
            self.start = start
            self.end = end
            self.data = self.get_data()
        else:
            self.data = pd.read_csv(os.path.join(Path.cwd().parent,'cleaned_data',self.symbol + '.csv'))

    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end)
        raw = raw.reset_index()
        raw.columns = ['Date','Close', 'High', 'Low', 'Open', 'Volume']
        raw = raw[['Date','Close','Open','Volume']]
        #raw = raw.loc[:,('Close', self.symbol)].rename('price').to_frame()
        #raw['return'] = np.log(raw['price']/raw['price'].shift(1))
        return raw
    
    def run_strategy(self, momentum):
        '''
        Backtests the trading strategy
        '''
        self.momentum = momentum
        data = self.data.copy().dropna()
        data['return'] = np.log(data['Close']/data['Close'].shift(1))
        data['signal'] = np.sign(data['return'].rolling(momentum).sum())
        data['position'] = data['signal'].shift(1) # executed position (lagged to avoid look-ahead)
        data.dropna(inplace = True)
        data['strategy'] = data['position'] * data['return']

        turnover = data['position'].diff().abs().fillna(0) # number of trades (2 when flipping sign)
        tc_fraction = self.tc / 100 if self.tc > 1 else self.tc
        if tc_fraction:
            if tc_fraction >= 1:
                raise ValueError("Transaction cost must be less than 100%")
            cost_log = -np.log1p(-tc_fraction) # log impact of cost on wealth
            data.loc[turnover > 0, 'strategy'] -= turnover * cost_log

        data['creturns'] =  self.amount * np.exp(data['return'].cumsum())
        data['cstrategy'] = self.amount * np.exp(data['strategy'].cumsum())
        self.results = data

        bh_factor = data['creturns'].iloc[-1] / self.amount
        strat_factor = data['cstrategy'].iloc[-1] / self.amount
        alpha = strat_factor - bh_factor
        
        return (f"buy&hold: {bh_factor - 1:.2%}, "
                f"strategy: {strat_factor - 1:.2%}, "
                f"alpha: {alpha:.2%}")
    
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = f"{self.symbol} | Momentum = {self.momentum} | TC = {self.tc} "
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = (10,6))
        plt.show()

    def max_drawdown(self): # Implementing Max Drawdown with Sliding Window
        l,r = 0,1
        max_dd = 0
        arr = self.results['cstrategy']
        while r < len(arr):
            if arr[l] > arr[r]:
                dd = arr[l] - arr[r]
                max_dd = max(max_dd,dd)
            else:
                l = r
            r += 1
        print(f'Max Drawdown:{max_dd}')
        return max_dd




#K = MomentumVectorBacktest('EURUSD=X','2023-12-01','2025-06-30',10000,0.0)
#K.run_strategy(3)
#K.plot_results()
#K.max_drawdown()
