"""
train.py - LSTM model training for dynamic pricing
"""
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from config.paths import MODEL_PATH
from model.pricing.preprocessing import PricePreprocessor, prepare_sequences, enrich_features


def build_lstm_model(input_shape):
    """
    Build LSTM model architecture
    
    Args:
        input_shape: (timesteps, features)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_model(df, sequence_length=7, epochs=50, batch_size=32):
    """
    Train LSTM model on historical pricing data
    
    Args:
        df: DataFrame with pricing history
        sequence_length: Number of time steps for LSTM
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        dict with training results
    """
    try:
        # Enrich features
        df = enrich_features(df)
        
        # Initialize preprocessor
        preprocessor = PricePreprocessor()
        
        # Fit and transform
        df_processed = preprocessor.fit_transform(df)
        
        # Prepare sequences
        X, y = prepare_sequences(df_processed, sequence_length=sequence_length)
        
        if len(X) == 0:
            return {
                "success": False,
                "message": "Not enough data to create sequences",
                "epochs": 0
            }
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Save model and preprocessor
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        model.save(f"{MODEL_PATH}/price_model.h5")
        
        with open(f"{MODEL_PATH}/preprocessor.pkl", 'wb') as f:
            pickle.dump(preprocessor, f)
        
        return {
            "success": True,
            "message": "Model trained successfully",
            "epochs": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "samples": len(X),
            "sequence_length": sequence_length
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "epochs": 0
        }