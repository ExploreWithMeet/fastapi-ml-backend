"""
preprocessing.py - Data preprocessing and feature engineering for LSTM model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime


class PricePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.categorical_features = ['demand_7d', 'time_of_day', 'season']  # ✅ REMOVED day_of_week
        self.numerical_features = ['current_price', 'rating_7d']  # ✅ REMOVED base_price, discount_perc
        self.binary_features = ['is_weekend', 'is_holiday', 'is_event', 'day_of_week']  # ✅ ADDED day_of_week here as it's 0-6  
   
    def fit(self, df):
        """Fit encoders and scaler on training data"""
        for feature in self.categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                le.fit(df[feature].astype(str))
                self.label_encoders[feature] = le
        
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            self.scaler.fit(df[numerical_cols])
        
        return self
    
    def transform(self, df):
        """Transform data using fitted encoders and scaler"""
        df = df.copy()
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature in df.columns and feature in self.label_encoders:
                df[feature] = self.label_encoders[feature].transform(df[feature].astype(str))
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Binary features are already 0/1
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform_price(self, scaled_price):
        """Convert scaled price back to original scale"""
        # Assuming current_price is first in numerical_features
        dummy = np.zeros((len(scaled_price), len(self.numerical_features)))
        dummy[:, 0] = scaled_price
        original = self.scaler.inverse_transform(dummy)
        return original[:, 0]


def prepare_sequences(df, sequence_length=7, target_col='current_price'):
    """
    Prepare sequences for LSTM training
    
    Args:
        df: Preprocessed dataframe
        sequence_length: Number of time steps to look back
        target_col: Column to predict
        
    Returns:
        X: Input sequences (samples, timesteps, features)
        y: Target values
    """
    feature_cols = [col for col in df.columns 
                if col not in [target_col, 'dish_id', 'timestamp', 'dt']]
    X, y = [], []
    
    # Group by dish_id to create sequences per dish
    for dish_id in df['dish_id'].unique():
        dish_data = df[df['dish_id'] == dish_id].sort_values('timestamp')
        
        if len(dish_data) < sequence_length + 1:
            continue
        
        dish_features = dish_data[feature_cols].values
        dish_target = dish_data[target_col].values
        
        for i in range(len(dish_data) - sequence_length):
            X.append(dish_features[i:i + sequence_length])
            y.append(dish_target[i + sequence_length])
    
    return np.array(X), np.array(y)


def enrich_features(df):
    """
    Add computed features from timestamp
    Used when predicting for new data
    """
    df = df.copy()
    
    if 'timestamp' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Time-based features
        df['is_weekend'] = df['dt'].dt.dayofweek >= 5
        df['day_of_week'] = df['dt'].dt.dayofweek  # ✅ CHANGED: Now returns 0-6 as int
        
        # Time of day
        df['hour'] = df['dt'].dt.hour
        df['time_of_day'] = df['hour'].apply(lambda h: 
            'MORNING' if 5 <= h < 12 else
            'NOON' if 12 <= h < 15 else
            'AFTERNOON' if 15 <= h < 20 else
            'NIGHT'
        )
        
        # Season
        df['month'] = df['dt'].dt.month
        df['season'] = df['month'].apply(lambda m:
            'WINTER' if m in [12, 1, 2] else
            'SUMMER' if m in [3, 4, 5, 6] else
            'MONSOON'
        )
        
        df.drop(['dt', 'hour', 'month'], axis=1, inplace=True)
    
    return df