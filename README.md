# Stock Price Trend Forecasting

## Introduction

LSTM (Long-Short-term memory) is a powerful RNN architecture, capable of learning long-term dependencies. 
In this project, we will predict stock price trend of the google stock price using keras, tensorflow.


## Methods

Summarized steps are followings:

1. Web scrape, normailize and reshape the data that fits to the architecture.
2. Construct the architecture with various number of nodes(units) and layers. 
In our example, 50 nodes and 5 layers (3 hidden layers) have been used, but combinations of 60, 120 nodes with 3, 4 layers have also been performed.
3. Predict 30 days stock trend using the 5 years of historical prices fed into the LSTM model.
4. Visualize/compare the actual trend and the predicted trend.
5. Fianlly, add two extra dimensions, open and close prices, in an attempt to get more accurate trend.

## Things to improve
I have also tried to forecast apple stock price using stock prices of other companies that are highly correlated with Apple, Foxconn and Samsung.
Foxconn is the no. 1 semiconductor supplier of Apple, and Samsung(also a semiconductor supplier)'s galaxy smartphones have very similar reputation and release as Apple' Iphone.
However, the model does not seem to capture the correlation. I would like to work further on this to understand why it does not work.
