import numpy as np
import pandas as pd
from scipy.optimize import brute
import yfinance as yf
import matplotlib.pyplot as plt
import typing
_UNCHANGED = object() # Unique Sentinel

class SMAVectorBacktest(object):
    '''
    Class for vectorised backtesting of a SMA based trading strategy
    
    '''
    def __init__(self,symbol:str,SMA1:int,SMA2:int,start:str,end:str):
        '''
        For now, yfinance API is used to retrieve historical data for backtest.
        start and end should be in 'yyyy-mm-dd' format as inputs for yfinance API.
        '''
        if SMA1 >= SMA2:
            raise ValueError("SMA1 must be smaller than SMA2.")

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
        raw = yf.download(self.symbol, start = self.start, end = self.end,auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            price = raw["Close"][self.symbol]
        else:
            price = raw["Close"]
        raw = price.rename("price").to_frame()
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
            
        if new1 >= new2:
            raise ValueError("new1 must be smaller than new2.")

        self.SMA1, self.SMA2 = new1, new2 # Committing both together (atomic switch in this object)

        self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self, return_factors:bool = False):
        '''
        Backtests the trading strategy.
        If intend to run optimise_parameters, we need to set return_factors to True to
        return numeric scores for the optimizer.
        Setting return_factors to False returns a formatted string with these scores.
        '''
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'] ,1 ,0)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace = True)
        data['creturns'] =  data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data

        bh_factor = data['creturns'].iloc[-1]       # gross performance of asset
        strat_factor = data['cstrategy'].iloc[-1]   # gross performance of strategy
        alpha_ret = (strat_factor - 1) - (bh_factor - 1) # out/under-performance of strategy

        if return_factors:
            return bh_factor, strat_factor, alpha_ret
        
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
        title = f"{self.symbol} | SMA1={self.SMA1} SMA2={self.SMA2}"
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = (10,6))
        plt.show()
    
    def update_and_run(self,SMA: tuple):
        '''
        Updates SMA Parameters and runs strategy again
        ''' 
        self.set_parameters(int(SMA[0]),int(SMA[1]))
        return self.run_strategy()
    
    def optimize_parameters(self, SMA1_range, SMA2_range):
        """
        Finds SMA1, SMA2 that maximize final strategy gross performance.
        SMA*_range = (start, end, step)
        """
        grid = (
            slice(SMA1_range[0], SMA1_range[1] + 1, SMA1_range[2]),
            slice(SMA2_range[0], SMA2_range[1] + 1, SMA2_range[2]),
        )

        def objective(SMA):
            s1, s2 = int(SMA[0]), int(SMA[1])
            if s1 >= s2:
                return 1e9  # penalty so invalid pairs are avoided

            self.set_parameters(s1, s2)
            _, strat_factor, _ = self.run_strategy(return_factors=True)
            return -float(strat_factor)  # negate because brute minimizes

        opt = brute(objective, grid, finish=None)
        best_s1, best_s2 = int(opt[0]), int(opt[1])

        # run once more at the optimum for reporting
        self.set_parameters(best_s1, best_s2)
        bh, strat, alpha = self.run_strategy(return_factors=True)

        return (f"opt: ({best_s1}, {best_s2}), "
                f"buy&hold: {bh - 1:.2%}, "
                f"strategy: {strat - 1:.2%}, "
                f"alpha: {alpha:.2%}")

    
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
        
    

    