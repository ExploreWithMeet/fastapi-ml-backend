import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from joblib import dump
from preprocessing import load_and_preprocess

MODEL_PATH = "model/price/price_model.pkl"

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(20))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_and_save():
    df, scaler = load_and_preprocess()

    # Prepare data for LSTM
    X = df[["sales", "demand", "is_weekend", "day_of_week"]].values.reshape(df.shape[0], 1, 4)
    y = df["price"].values

    model = build_model((1, X.shape[2]))
    model.fit(X, y, epochs=30, batch_size=1, verbose=2)

    dump((model, scaler), MODEL_PATH)   
    return {"message":f"Model trained and saved at {MODEL_PATH}"}
