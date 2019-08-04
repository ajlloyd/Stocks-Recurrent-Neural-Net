import numpy as np
import pandas as pd
from pandas_datareader import data
import datetime as dt

access_key = "-KqTJKwCissxT3a7vQ2T"
start = dt.datetime(2009, 01, 01)
end = dt.datetime(2014, 01, 01)
df = data.DataReader("BAC", "quandl", start, end, access_key=access_key)
print(df.tail())

print("test")
