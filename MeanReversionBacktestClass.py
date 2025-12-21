import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class MRVectorBacktest():
    '''
    Class for backtest of a mean reversion trading strategy
    '''
    def __init__(self,symbol:str,window:int,entry:int,exit:int,start:str,end:str,ann_factor:int=252):
        self.symbol = symbol
        self.start = start 
        self.end = end 
        self.results = None
        self.data = self.get_data()
        self.pos = None
        self.ann = ann_factor
        self.window = window
        self.entry = entry
        self.exit = exit
        
    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            price = raw["Close"][self.symbol]
        else:
            price = raw["Close"]

        raw = price.rename("price").to_frame()
        raw['return'] = np.log(raw['price']/raw['price'].shift(1))
        return raw
    
    def build_positions(self,entry:float,exit:float,window:int):
        '''
        Building a position series where elements in {-1,0,1}, shifted(1)
        '''

        if window <= 1:
            raise ValueError("window must be > 1")
        if entry <= 0 or exit < 0:
            raise ValueError("entry must be > 0 and exit must be >= 0")
        if entry <= exit:
            raise ValueError("entry should be > exit")

        
        price = self.data['price'] # price is a pd.Series
        sma = price.rolling(window).mean()
        dist = price - sma
        sigma = dist.rolling(window).std()
        mu = dist.rolling(window).mean()
        eps = 1e-12
        z = dist / sigma.replace(0,np.nan) # pd.Series of z score

        long_entry = z < -entry # boolean pd.Series
        short_entry = z > entry
        flat_exit = z.abs() < exit
        

        pos = np.where(flat_exit, 0, np.nan)
        pos = np.where(long_entry, 1, pos)
        pos = np.where(short_entry, -1, pos)
        
        pos =  pd.Series(pos, index = price.index).ffill().shift(1).fillna(0) # pos pd.Series, shifted by 1
        
        self.pos = pos
        return pos

    def score(self,pos: pd.Series):
        '''
        Takes in positions vector and output scoring metrics
        '''
        returns = self.data['return']
        strat = (pos * returns).copy()
        

        creturns = returns.cumsum().apply(np.exp) 
        cstrategy = strat.cumsum().apply(np.exp)

        self.data['creturns'] = creturns
        self.data['cstrategy'] = cstrategy

        mean = strat.mean() * self.ann
        vol = strat.std() * np.sqrt(self.ann) + 1e-16

        sharpe = mean/vol
        
        crets = float(creturns.iloc[-1])
        cstrat = float(cstrategy.iloc[-1])
        cagr = float(cstrategy.iloc[-1] ** (self.ann/max(len(cstrategy),1)) -1)
        alpha_ret = (cstrat - 1) - (crets - 1)

        return {
        "sharpe": float(sharpe),
        "cagr": f"{cagr:.2%}",
        "cstrat": f"{(cstrat - 1):.2%}",
        "crets": f"{(crets - 1):.2%}",
        "alpha": f"{alpha_ret:.2%}"
        }
                    
    def run_strategy(self): ## single run of backtest

        pos = self.build_positions(entry = self.entry,exit = self.exit,window = self.window)
        self.results = self.score(pos) | {'params':{'entry': self.entry, 'exit': self.exit, 'window':self.window}} #| is dict union/merge operator

        return self.results
        
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        title = f"{self.symbol} | window: {self.window} | entry_condition: {self.entry} | exit_condition: {self.exit} "
        pd.DataFrame(self.data[['creturns', 'cstrategy']]).plot(title = title, figsize = (10,6))
        plt.show()


