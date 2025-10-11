import numpy as np
import pandas as pd
from scipy.optimize import brute
import yfinance as yf
import matplotlib.pyplot as plt
_UNCHANGED = object() # Unique Sentinel

class SMAVectorBacktester(object):
    '''
    Class for vectorised backtesting of a SMA based trading strategy
    
    '''
    def __init__(self,symbol,SMA1,SMA2,start,end):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.data = self.get_data()

    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end)
        raw = raw.loc[:,('Close', self.symbol)].rename('price').to_frame()
        raw['return'] = np.log(raw['price']/raw['price'].shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        self.data = raw
        return self.data
        
    
    def set_parameters(self,SMA1 = _UNCHANGED, SMA2 = _UNCHANGED):
        '''
        Updates SMA parameters and generates SMA Time Series
        '''
        new1 = self.SMA1 if SMA1 is _UNCHANGED else SMA1     
        new2 = self.SMA2 if SMA2 is _UNCHANGED else SMA2 

        for v,name in [(new1,"SMA1"), (new2,"SMA2")]:
            if v is not None and (not isinstance(v, int) or v <= 0):
                raise ValueError(f"{name} must be a positive integer.")

        self.SMA1, self.SMA2 = new1, new2 # Committing both together (atomic switch in this object)

        self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        '''
        Backtests the trading strategy
        '''
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'] ,1 ,-1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace = True)
        data['creturns'] =  data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
    
        grossperf = data['cstrategy'].iloc[-1] # gross performance of strategy
        operf = grossperf - data['creturns'].iloc[-1] # out/under-performance of strategy
        
        return round(grossperf,2), round(operf,2)
    
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = f"{self.symbol} | SMA1={self.SMA1} SMA2={self.SMA2}"
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = (10,6))
        plt.show()
    
    def update_and_run(self,SMA: tuple):
        '''
        Updates SMA Parameters and returns negative absolute performance for minimization algorithm
        ''' 
        self.set_parameters(int(SMA[0]),int(SMA[1]))
        return -self.run_strategy()[0]
    
    def optimize_parameters(self, SMA1_range, SMA2_range):
        '''
        Finds global maximum given the SMA parameter ranges
        '''
        grid = (
            slice(SMA1_range[0], SMA1_range[1] + 1, SMA1_range[2]),
            slice(SMA2_range[0], SMA2_range[1] + 1, SMA2_range[2])
        )

        opt = brute(self.update_and_run, grid, finish = None)
        return f"opt: {opt}, grossperf: {-self.update_and_run(opt)}"


L = SMAVectorBacktester('EURUSD=X', 42,252,'2023-12-01','2025-06-30')
L.run_strategy()
L.optimize_parameters((30,56,4),(200,300,4))