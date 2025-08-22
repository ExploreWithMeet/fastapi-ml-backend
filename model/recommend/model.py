import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

MODEL_PATH = Path("model/price/price_model.pkl")

def predict_price(dish_id: str, recent_data: pd.DataFrame) -> float:
    """
    Predict next price for a given dish_id using trained model.
    
    Parameters
    ----------
    dish_id : str
        The ID of the dish to predict for
    recent_data : pd.DataFrame
        Must contain ['dish_id','timestamp','sales','demand'] at minimum
    
    Returns
    -------
    float : predicted price
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Please run training first.")

    # Load trained model & scaler
    model, scaler = load(MODEL_PATH)

    # Filter data for the given dish
    dish_data = recent_data[recent_data["dish_id"] == dish_id].copy()
    if dish_data.empty:
        raise ValueError(f"No data found for dish_id {dish_id}")

    # Convert timestamp + create features
    dish_data["timestamp"] = pd.to_datetime(dish_data["timestamp"], unit="ms", errors="coerce")
    dish_data["day_of_week"] = dish_data["timestamp"].dt.dayofweek
    dish_data["is_weekend"] = dish_data["day_of_week"].isin([5, 6]).astype(int)

    # Define features (must match training features!)
    features = ["sales", "demand", "is_weekend", "day_of_week"]
    if not set(features).issubset(dish_data.columns):
        raise ValueError(f"Missing required columns: {set(features) - set(dish_data.columns)}")

    # Scale
    X_new = scaler.transform(dish_data[features].values)

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_new = X_new.reshape(X_new.shape[0], 1, X_new.shape[1])

    # Predict
    predictions = model.predict(X_new, verbose=0)

    # Return last predicted value
    return float(predictions[-1][0])
