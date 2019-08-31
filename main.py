
import quandl

tickers = ["AAPL", "MSFT", "NVDA", "HPQ", "GOOGL", "INTC", "AMZN"]

for ticker in tickers:
    data = quandl.get(f"WIKI/{ticker}", start_date="2014-3-27", end_date="2018-3-27")
    data.to_csv(f"./tickers/{ticker}.csv")
