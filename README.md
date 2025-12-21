# Backtest-Classes

This repo contains a series of backtest classes (via vectorisation) implementing three of the most widely used trading strategies:
1. Simple Moving Average
2. Mean Reversion 
3. Momentum

## General Design for Vectorised Backtest Classes
This section describes the general backtesting workflow used across the classes. When .get_data() is called, price data is downloaded from the yfinance API for the specified start and end dates and stored in a pandas DataFrame. We then compute a *return* column using log returns: return = log(Close_t/Close_t-1) (We work with log returns instead of simple returns thoughout the backtest). Depending on the strategy, we generate a daily *postion* signal that represents the exposure helo over the period (+1 for long, -1 for short and 0 for no position). To avoid look-ahead bias, the *position* series is shifted by one day so that today's trade uses yesterday's signal. Strategy returns are then computed as *strategy* = *position.shift(1) * return*. To convert cumulative log returns into cumulative simple returns, we take the exponent of the cumulative sum of *return* and *strategy*. *creturns* = *exp(return.cumsum())* and *cstrategy* = exp(strategy.cumsum())* The 'buy and hold' performance (total return of the asset) is given by the last value of the *creturns* series *df['creturns'].iloc(-1)* while the strategy's return is given by the last value of the *cstrategy* series *df['cstrategy].iloc(-1)*.

*The implementation and variable naming for MeanReversionBacktestClass.py is slightly different as I was experimenting witha diff implementation but the underlying logic design logic still remains.*

*The implementation design, variable naming etc. might slightly differ from class to class as since my intended goal was to explore and experimengt with different implementation designs.*

## Simple Moving Average -SMA
This class simulates the backtested performance of a Simple Moving Average trading strategy. The intuition for this strategy is to go long when short term performance beats longer term performance and to go flat (no position) vice versa. Two simple moving average readings are taken based on rolling mean of past prices (This ensures no data leakage - we are not making decisions based on data from the future). SMA1 denotes a **shorter** lookback period while SMA2 denotes a **longer** lookbak period. A long position (+1 signal) if taken when when SMA1 crosses over SMA2 and a flat position (0 signal) is taken when SMA1 falls below SMA2.

***Siganl logic as follows***:
SMA1: Simple moving avg of price over shorter lookback period as a measure of short term performance
SMA2 :Simple moving avg of price over longer lookback period as a measure of longer term performance
+1 signal (long) when SMA1 > SMA2
0 signal (short) when SMA1 <= SMA2

## Mean Reversion
This class simulates the backtested performance of a Mean Reversion trading strategy. The intuition for this strategy is that prices tend to fluctuate and thus revert to its rolling mean over a period of time. For every day in the backtested dataset, we build a distance metric which is defined by the rolling mean of price for a certain preceding period (specified by user) subtracted from the day's current price. We then normalise this distance metric by dividing by the rolling standard deviation of distance over the same period (can be different as well). This gives us a standardised distance metric which I denote z score. A long position (+1 signal) is taken when the z score is below a certain threshold while a short position is taken while the z score is above a certain threshold. If the current z score is inbetween the upper and lower thresholds (you can visualise a band), no position is taken (0 signal).

***Signal logic as follows***:
Every calculation day, distance = price - SMA over a user specified period
Normalise this distance using (distance - SMA of distance)/simple moving standard deviation of distance
+1 signal (long) if z < lower threshold         
-1 signal (short) if z > upper threshold
0 signal (flat postiion) if z >= lower threshold and z <= upper threshold
*thresholds are integers specified by the user*

## Momentum
This class simulates the backtested performance of a momentum based trading strategy. The intuition for this strategy is to test if strong short term time series momentum should continue in the future (at least in the very short term). For everyday in the backtest dataset, we compute a daily log return metric by taking the natural log of the closing price of the current day divided by the closing price of the previous day. The user inputs a 'momentum' variable which denotes the length of our lookback period. If the sign of the rolling sum of log returns (which is equivalent to the log of the product of simeple gross returns) over 'momentum' lookback window variable is positive , we take a long position (+1 signal) and if it yis negative, we take a short position (-1 signal).

For this class, I also decided to experiment with modelling transaction costs, applied whenever the position changes (switching between long and short).

***Signal logic as follows***
Every calculation day, take the rolling sum of the log returns over the 'momentum' variable lookback window
If the sign of the value is positive, we take a long position (+1 signal) on the current day
If the sign of this value is negative, we take a short position (-1 signal) on the current day
Transactions costs are also computed in the log world. For each trading action (going from a long to short posiition or vice versa),
we subtract the log cost from the strategy's return


## Use Cases for Backtest Classes
One practical use case for these backtest classes is building an asset screener. For example, you can run the same strategy across a basket of equities within the same industry or economic classification, then compare results to identify which assets have historically performed best under that strategy.

