# !pip install yahoo_fin
# !pip install yfinance
# !pip install yahoofinancials


import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
# import yahoo_fin.stock_info as si
import yfinance as yf
from app_v3 import main
# from yahoofinancials import YahooFinancials

class Model():
    '''
    A model for predicting the stock price
    '''
    def __init__(self):
        '''
        starting out 
        '''
        
        
    def extract_data(self, ticker):
        dl_df= yf.download(ticker, start='2021-01-01', 
                    end='2022-04-01', progress=False)
    
        self.ticker_set = pd.DataFrame(dl_df)
        self.ticker_set.reset_index(inplace = True)
        return self.ticker_set


    def reshape(self):

        stock_top_df = pd.read_csv("dataframes_top.csv")
        dataset_total = pd.concat((stock_top_df['Open'], self.ticker_set['Open']), axis = 0)
        # Extract Stock Prices for Test time period, plus 60 days previous
        inputs = dataset_total[len(dataset_total) - len(self.ticker_set) - 60:].values
        # 'reshape' function to get it into a NumPy format
        inputs = inputs.reshape(-1,1)
        # Scaling the input
        scaler = MinMaxScaler(feature_range = (0, 1))
        inputs = scaler.fit_transform(inputs)

        X_test = []

        for i in range(60, len(inputs)):
            X_test.append(inputs[i-60:i, 0])

        X_test = np.array(X_test)
        # Making the input in 3D format
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test
# Graph Function
def make_graph(self, ticker, output_file='predictions_pic.svg'):
  date = self.ticker_set['Date']
  Last314Days = date.tail(314)
  real_df = self.ticker_set['Open']
  real_df.columns = ['Real']
  predicted_df = main.output
  predicted_df.columns = ['Predicted']
  line_df = pd.concat([Last314Days, real_df, predicted_df], axis=1)
  fig = go.Figure([
                 go.Scatter(
                     x=line_df['Date'],
                     y=line_df[f'Real {ticker}'],
                    showlegend=True,
                    name=f'Real {ticker}'
                 ),
                 go.Scatter(
                     x=line_df['Date'],
                     y=line_df[f'Predicted {ticker}'],
                    showlegend=True,
                    name=f'Predicted {ticker}'
                 )
  ])
  fig.update_layout(
                    yaxis_title = 'Stock Prices (USD)',
      xaxis_title='Dates',
      title=f'Real and Predicted {ticker} Stock Prices',
      hovermode="x"
  )
  fig.write_image(output_file, width=800)
  return fig


                
