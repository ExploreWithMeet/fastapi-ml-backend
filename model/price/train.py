import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump
from config.paths import PRICING_MODEL_PATH
from model.price.preprocessing import preprocess_data
from keras import Sequential
from keras.layers import Dense, LSTM

def train_model(df: pd.DataFrame):
    # df, encoders = preprocess_data(df)
    X = df.drop(columns=["predicted_price"])
    y = df["predicted_price"]

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], 1)))
    # model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    dump(model, PRICING_MODEL_PATH)
    return model

