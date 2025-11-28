import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class MRVectorBacktest():
    '''
    Class for backtest of a mean reversion trading strategy
    '''
    def __init__(self,symbol,start,end,ann_factor=252):
        self.symbol = symbol
        self.start = start 
        self.end = end 
        self.results = None
        self.data = self.get_data()
        self.pos = None
        self.ann = ann_factor
        self.window = None
        self.entry = None
        self.exit = None
        
    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end)
        raw = raw.loc[:,('Close', self.symbol)].rename('price').to_frame()
        raw['return'] = np.log(raw['price']/raw['price'].shift(1))
        return raw
    
    def build_positions(self,entry:float,exit:float,window:int):
        '''
        Building a position series where elements in {-1,0,1}, shifted(1)
        '''
        self.window = window
        self.entry = entry
        self.exit = exit
        price = self.data['price'] # price is a pd.Series
        sma = price.rolling(window).mean()
        dist = price - sma
        sigma = dist.rolling(window).std()
        mu = dist.rolling(window).mean()
        eps = 1e-12
        z = (dist - mu) / sigma.replace(0,eps) # pd.Series of z score
        print(f'z:{z}')
        print(type(z))

        long_entry = z < -entry # boolean pd.Series
        short_entry = z > entry
        flat_exit = z.abs() < exit
        print(f'long_entry:{long_entry}') 
        print(type(long_entry))

        pos = np.where(flat_exit, 0, np.nan)
        pos = np.where(long_entry, 1, pos)
        pos = np.where(short_entry, -1, pos)
        
        pos =  pd.Series(pos, index = price.index).ffill().shift(1).fillna(0) # pos pd.Series, shifted by 1
        
        self.pos = pos
        return pos

    def score(self,pos: pd.Series):
        '''
        Takes in positions vector and output scoring metrics   #Make this  ahelper function??
        '''
        returns = self.data['return']
        strat = (pos * returns).copy()
        #print(f'strat:{strat}')

        cumrets = returns.cumsum().apply(np.exp) 
        equity = strat.cumsum().apply(np.exp)

        self.data['cumrets'] = cumrets
        self.data['equity'] = equity

        mean = strat.mean() * self.ann
        vol = strat.std() * np.sqrt(self.ann) + 1e-16

        sharpe = mean/vol
        
        crets = cumrets.iloc[-1]
        cstrat = equity.iloc[-1]
        cagr = equity.iloc[-1] ** (self.ann/max(len(equity),1)) -1

        return {"sharpe":float(sharpe),                    
                "cagr": float(cagr),
                "cstrat":float(cstrat),
                "crets":float(crets)
                }
                    
    def run_strategy(self): ## single run of backtest

        pos = self.build_positions(self.entry,self.exit,self.window)
        self.results = self.score(pos) | {'params':{'entry': self.entry, 'exit': self.exit, 'window':self.window}} #| is dict union/merge operator

        return self.results
        
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = f"{self.symbol} | window: {self.window} | entry_condition: {self.entry} | exit_condition: {self.exit} "
        pd.DataFrame(self.data[['cumrets', 'equity']]).plot(title = title, figsize = (10,6))
        plt.show()


#-------------------------test-------------#
symbol = 'EURUSD=X'
start = '2023-12-01'
end = '2025-06-30'
P = MRVectorBacktest(symbol,start, end)
P.get_data()
P.build_positions(0.9,0.5,25)
P.run_strategy()
P.results
P.plot_results()
P.data