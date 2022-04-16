import pandas as pd
import yahoo_fin.stock_info as si
import yfinance as yf
from yahoofinancials import YahooFinancials
import cufflinks as cf
cf.set_config_file(offline=True)

def scrape():

    sp500 = si.tickers_sp500()
    sp500_df = pd.DataFrame()

    if user_input = sp500:
        sp500_dl = yf.download(sp500, progress=False)
        sp500_dl['Ticker'] = ticker
        sp500_df = dow_df.append(dow_dl)
    else:
        return "Not in index"