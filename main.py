
import quandl
import pandas as pd
import numpy as np
import os

print(os.getcwd())

#-------------------------------------------------------------------------------
#Globals:

TICKERS = ["AAPL", "MSFT", "NVDA"] #, "HPQ", "GOOGL", "INTC", "AMZN"]
TO_PREDICT = "NVDA"
FUTURE = 3    #close column will be shifted 5 days into future to produce labels
SEQUENCE_LEN = 25


#-------------------------------------------------------------------------------
def pull_csv(tickers):
    for ticker in TICKERS:
        data = quandl.get(f"WIKI/{ticker}", start_date="2014-3-27", end_date="2018-3-27")
        data.to_csv(f"./tickers/{ticker}.csv")
#pull_csv(TICKERS)

def buy_or_sell(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def make_targets():
    main_df = pd.DataFrame()
    csv_folder = "./tickers/"
    for csv in os.listdir(csv_folder):
        ticker_path = os.path.join(csv_folder, csv)
        df = pd.read_csv(ticker_path)
        df.set_index("Date", inplace=True)
        df = df[["Close", "Volume","Ex-Dividend", "Adj. Close","Adj. Volume"]]
        ticker_name = csv.split(".")[0]
        df.rename(columns={"Close" : f"Close_{ticker_name}",
                           "Volume" : f"Volume_{ticker_name}",
                           "Ex-Dividend" : f"Div_{ticker_name}",
                           "Adj. Close" : f"Adj_Close_{ticker_name}",
                           "Adj. Volume" : f"Adj_Volume_{ticker_name}"
                           }, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df[f"Future_Close_{TO_PREDICT}"] = main_df[f"Adj_Close_{TO_PREDICT}"].shift(FUTURE)
    main_df["Target"] = list(map(buy_or_sell, main_df[f"Adj_Close_{TO_PREDICT}"], main_df[f"Future_Close_{TO_PREDICT}"]))
    return main_df.drop(f"Future_Close_{TO_PREDICT}",1)
df = make_targets()

def train_validation_split():
    pass
