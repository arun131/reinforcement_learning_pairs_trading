import numpy as np
import random
from collections import deque
from pair_trading_env import PairtradingEnv
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl 
import pandas as pd 

# stocklist = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'IOC.NS', 'INFY.NS','KOTAKBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS']
stocklist = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
stocktickers = ' '.join(stocklist)

data = yf.download(tickers = stocktickers, start="2024-01-01", end="2024-02-04", interval = '5m')
data_close = data['Adj Close']


columnchange = []
for stock in data_close.columns:
    name = stock+'change'
    columnchange.append(name) 
    data_close[name] = data_close[stock] - data_close[stock].shift(1)

CorrDict = {}
for i in columnchange :
    for j in columnchange :
        if i != j  and (i,j) not in CorrDict and (j,i) not in CorrDict:
            CorrDict[(i,j)] = data_close[i].corr(data_close[j])
pair = list(max(CorrDict))
pair.append(pair[0][:-6])
pair.append(pair[1][:-6]) 
dataremain = data_close[pair] 
pair_names = [n for n in pair if "change" not in n]
stock1 = dataremain[pair_names[0]]
stock2 = dataremain[pair_names[1]]

# dataremain[pair_names].plot(secondary_y=pair_names[1])
# plt.show()

pair_env = PairtradingEnv(stock1 = stock1, stock2 = stock2, cash = 100000, max_trade_period= 1000)