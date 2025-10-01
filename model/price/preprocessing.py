import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# def merge_events(df: pd.DataFrame, events_df: pd.DataFrame):
#     """Add event flags from events.csv"""
#     if events_df.empty:
#         df["is_event_csv"] = False
#         return df

#     events_df["dt"] = pd.to_datetime(events_df["timestamp"], unit="ms")
#     df["is_event_csv"] = df["dt"].isin(events_df["dt"].dt.floor("D"))
#     return df


# def preprocess_data(df: pd.DataFrame, timesteps: int = 10):
#     """Preprocess training data"""
#     target_col = "current_price"
#     feature_cols = [c for c in df.columns if c not in ["predicted_price", "timestamp", "iso_timestamp", "dt", target_col]]

#     X_df = df[feature_cols].copy()
#     y = df[target_col].values

#     # Encode categoricals
#     for col in X_df.select_dtypes(include=["object", "bool"]).columns:
#         le = LabelEncoder()
#         X_df[col] = le.fit_transform(X_df[col].astype(str))

#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X_df)

#     X, Y = [], []
#     for i in range(len(X_scaled) - timesteps):
#         X.append(X_scaled[i:i+timesteps])
#         Y.append(y[i+timesteps])
#     return np.array(X), np.array(Y), scaler, feature_cols


# def preprocess_for_prediction(df: pd.DataFrame, scaler, feature_cols, timesteps: int):
#     """Use saved scaler + features to preprocess new data for prediction"""
#     X_df = df[feature_cols].copy()

#     for col in X_df.select_dtypes(include=["object", "bool"]).columns:
#         X_df[col] = X_df[col].astype("category").cat.codes

#     X_scaled = scaler.transform(X_df)

#     X = []
#     for i in range(len(X_scaled) - timesteps):
#         X.append(X_scaled[i:i+timesteps])

#     return np.array(X)

def preprocess_data():
    pass