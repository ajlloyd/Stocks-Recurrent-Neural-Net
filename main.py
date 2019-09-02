
import quandl
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, GRU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

pd.options.mode.chained_assignment = None
print(os.getcwd())

#-------------------------------------------------------------------------------
#Globals:

TICKERS = ["AAPL", "MSFT", "NVDA", "HPQ", "GOOGL", "INTC", "AMZN"]
TO_PREDICT = "NVDA" # Ticker price to be predicted
FUTURE = 3          # Adj Close column will be shifted 5 days into future to produce labels
SEQUENCE_LEN = 10   # Length of one "sequence feature" - (10 x (n_tickers(7) * n_columns(2)))
EPOCHS = 10         # Keras Epochs
BATCH = 32          # Sequence Features in a keras batch

#-------------------------------------------------------------------------------
def pull_csv(tickers):
    ### Pull tickers from QUANDL
    for ticker in TICKERS:
        data = quandl.get(f"WIKI/{ticker}", start_date="1998-3-27", end_date="2018-3-27")
        data.to_csv(f"./tickers/{ticker}.csv")
pull_csv(TICKERS)

def buy_or_sell(current, future):
    ### function to map to feature column using map()
    if float(future) > float(current):
        return 1
    else:
        return 0

def make_targets():
    ### Produce target (label) column based on the future price (0 sell, 1 buy)
    main_df = pd.DataFrame()
    csv_folder = "./tickers/"
    for csv in os.listdir(csv_folder):
        ticker_path = os.path.join(csv_folder, csv)
        df = pd.read_csv(ticker_path)
        df.set_index("Date", inplace=True)
        df = df[["Close", "Volume", "Adj. Close","Adj. Volume"]]
        ticker_name = csv.split(".")[0]
        df.rename(columns={"Close" : f"Close_{ticker_name}",
                           "Volume" : f"Volume_{ticker_name}",
                           "Adj. Close" : f"Adj_Close_{ticker_name}",
                           "Adj. Volume" : f"Adj_Volume_{ticker_name}"}, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df[f"Future_Close_{TO_PREDICT}"] = main_df[f"Adj_Close_{TO_PREDICT}"].shift(FUTURE)
    main_df["Target"] = list(map(buy_or_sell, main_df[f"Adj_Close_{TO_PREDICT}"],
                                  main_df[f"Future_Close_{TO_PREDICT}"]))
    return main_df.drop(f"Future_Close_{TO_PREDICT}",1)
df = make_targets()

def OOS_data(df, split_pct):
    ### Split Data into test and val data
    df_index = df.index.values
    index_len = len(df_index)
    val_size = int(split_pct*index_len)
    cuttoff_index = df_index[-val_size]
    val_df = df[(df.index >= cuttoff_index)]
    main_df = df[(df.index < cuttoff_index)]
    return main_df, val_df
main_df = OOS_data(df, 0.1)[0]
val_df = OOS_data(df, 0.1)[1]

def normalisation(df):
    ### StandardScaler Normalises Data (Mean 0, var 1):
    scaler = StandardScaler()
    for col in df.columns:
        if col != "Target":
            df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
            df.dropna(inplace=True)
    return df

def balanced_sequential_data(df):
    ### Producing Sequential Data:
    sequence_data = []
    previous_days = deque(maxlen=SEQUENCE_LEN)
    for row in df.values:
        row_i = []
        for value in row[:-1]:
            row_i.append(value)
        previous_days.append(row_i)
        if len(previous_days) == SEQUENCE_LEN:
            sequence_data.append([np.array(previous_days), int(row[-1])])
    random.shuffle(sequence_data)
    ### Producing Balanced Sequential Data:
    buys = []
    sells = []
    for feature, label in sequence_data:
        if label == 1:
            buys.append([feature, label])
        if label == 0:
            sells.append([feature, label])
    smaller = min(len(buys), len(sells))
    buys = buys[:smaller]
    sells = sells[:smaller]
    balanced_data = buys + sells
    random.shuffle(balanced_data)
    ### Split Balanced Sequential Data into Features and Labels
    X = []
    y = []
    for feature, label in balanced_data:
        X.append(feature)
        y.append(label)
    return np.array(X), np.array(y)
X_train, y_train = balanced_sequential_data(normalisation(main_df))
X_val, y_val = balanced_sequential_data(normalisation(val_df))
#-------------------------------------------------------------------------------

model_name = f"{TO_PREDICT}-{SEQUENCE_LEN}-SEQ-{FUTURE}-DAYS-{int(time.time())}"
input = X_train.shape[1:] # The shape of one feature (10 x 14) (where 10 is the SEQUENCE_LEN and 14 is n_columns(2) * n_tickers(7))
### Layer 1:
model = Sequential()
model.add(GRU(128, input_shape=(input), return_sequences=True, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
### Layer 2:
model.add(GRU(128, input_shape=(input), return_sequences=True, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
### Layer 3:
model.add(GRU(128, input_shape=(input), activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
### Layer 4:
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
### Output:
model.add(Dense(2, activation="softmax"))
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss="sparse_categorical_crossentropy",
               optimizer=opt,
               metrics=["accuracy"])
tensorboard = TensorBoard(log_dir=f".\logs\\{model_name}")
model.fit(X_train, y_train,batch_size=BATCH,epochs=EPOCHS,
                    validation_data=(X_val, y_val),callbacks=[tensorboard])
