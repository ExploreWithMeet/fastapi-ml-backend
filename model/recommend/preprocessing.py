import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

EVENTS_PATH = Path("model/price/event.csv")

def save_daily_data(row: dict):
    """Append a daily row (dict) into event.csv"""
    if EVENTS_PATH.exists():
        df = pd.read_csv(EVENTS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EVENTS_PATH, index=False)

def load_and_preprocess():
    """Load all events & preprocess for training"""
    if not EVENTS_PATH.exists():
        raise FileNotFoundError("No event.csv found")

    df = pd.read_csv(EVENTS_PATH)

    # Example feature engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    # Handle missing values
    df = df.fillna(0)

    # Scaling features (for LSTM input)
    scaler = MinMaxScaler()
    features = ["sales", "demand", "is_weekend", "day_of_week"]
    df[features] = scaler.fit_transform(df[features])

    return df, scaler
