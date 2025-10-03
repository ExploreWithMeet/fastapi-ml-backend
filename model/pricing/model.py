"""
model.py - Price prediction using trained LSTM model
"""
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from config.paths import MODEL_PATH
from model.pricing.preprocessing import enrich_features


def predict_price(df, sequence_length=7):
    """
    Predict price for given input data
    
    Args:
        df: DataFrame with current data (must have historical sequences)
        sequence_length: Number of time steps model was trained on
        
    Returns:
        float: Predicted price
    """
    try:
        # ✅ CHECK IF MODEL EXISTS FIRST
        if not (MODEL_PATH / "price_model.h5").exists():
            print("Model not found, returning current price")
            return float(df['current_price'].iloc[-1]) if 'current_price' in df.columns else 0.0
        
        # Load model and preprocessor
        model = load_model(MODEL_PATH / "price_model.h5")  # ✅ FIXED: Use Path object
        
        with open(MODEL_PATH / "preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Enrich features
        df = enrich_features(df)
        
        # Transform
        df_processed = preprocessor.transform(df)
        
        # For prediction, we need last 'sequence_length' records
        if len(df_processed) < sequence_length:
            df_processed = pd.concat([df_processed] * sequence_length, ignore_index=True)
        
        # Take last sequence_length rows
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['current_price', 'dish_id', 'timestamp', 'dt']]  # ✅ FIXED: Added exclusions
        
        X = df_processed[feature_cols].tail(sequence_length).values
        X = X.reshape(1, sequence_length, len(feature_cols))
        
        # Predict
        predicted_scaled = model.predict(X, verbose=0)[0][0]
        
        # Inverse transform
        predicted_price = preprocessor.inverse_transform_price([predicted_scaled])[0]
        
        return float(predicted_price)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return float(df['current_price'].iloc[-1]) if 'current_price' in df.columns else 0.0

def predict_batch_prices(df, sequence_length=7):
    """
    Predict prices for multiple dishes
    
    Args:
        df: DataFrame with multiple dish records
        sequence_length: Sequence length
        
    Returns:
        dict: {dish_id: predicted_price}
    """
    predictions = {}
    
    for dish_id in df['dish_id'].unique():
        dish_data = df[df['dish_id'] == dish_id].sort_values('timestamp')
        predicted_price = predict_price(dish_data, sequence_length)
        predictions[dish_id] = predicted_price
    
    return predictions