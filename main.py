
import quandl
import pandas as pd
import numpy as np
import os

print(os.getcwd())

#-------------------------------------------------------------------------------
#Globals:

TICKERS = ["AAPL", "MSFT", "NVDA", "HPQ", "GOOGL", "INTC", "AMZN"]
TO_PREDICT = ["NVDA"]
FUTURE = 5    #close column will be shifted 5 days into future to produce labels
SEQUENCE_LEN = 25


#-------------------------------------------------------------------------------
def pull_csv(tickers):
    for ticker in TICKERS:
        data = quandl.get(f"WIKI/{ticker}", start_date="2014-3-27", end_date="2018-3-27")
        data.to_csv(f"./tickers/{ticker}.csv")
#pull_csv(TICKERS)

def targets(future_period, sequence_length):
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

    

targets(3, 3)
