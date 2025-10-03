import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = "pricing_data.csv"
MODEL_DIR = Path("model/pricing")
SEQUENCE_LENGTH = 30
EPOCHS = 75
BATCH_SIZE = 32
TEST_SIZE = 0.2

class PricePreprocessor:
    """Handle all data preprocessing and feature engineering"""
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.categorical_features = ['demand_7d', 'time_of_day', 'season','day_of_week']
        self.numerical_features = ['current_price', 'rating_7d']
        self.binary_features = ['is_weekend', 'is_holiday', 'is_event']

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

        # Binary features remain as integers
        for feature in self.binary_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(int)

        return df

    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)

    def inverse_transform_price(self, scaled_price):
        """Convert scaled price back to original scale"""
        if isinstance(scaled_price, (int, float)):
            scaled_price = [scaled_price]

        scaled_price = np.array(scaled_price).flatten()
        dummy = np.zeros((len(scaled_price), len(self.numerical_features)))
        dummy[:, 0] = scaled_price
        original = self.scaler.inverse_transform(dummy)
        return original[:, 0]

def prepare_sequences(df, sequence_length=7, target_col='current_price'):
    """Prepare sequences for LSTM training"""
    if 'dish_id' not in df.columns:
        raise ValueError("dish_id column is required")

    feature_cols = [col for col in df.columns
                   if col not in [target_col, 'dish_id', 'timestamp', 'dt', 'event_name', 'rest_id']]

    print(f"Using {len(feature_cols)} features for each sequence")

    X, y = [], []
    skipped = 0

    for dish_id in df['dish_id'].unique():
        dish_data = df[df['dish_id'] == dish_id].sort_values('timestamp')

        if len(dish_data) < sequence_length + 1:
            skipped += 1
            continue

        dish_features = dish_data[feature_cols].values
        dish_target = dish_data[target_col].values

        for i in range(len(dish_data) - sequence_length):
            X.append(dish_features[i:i + sequence_length])
            y.append(dish_target[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    if skipped > 0:
        print(f"Skipped {skipped} dishes with insufficient data")

    print(f"Sequences: {len(X)}")
    print(f"Shape: X={X.shape}, y={y.shape}")

    return X, y

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation=LeakyReLU(alpha=0.1)),
        Dense(8, activation=LeakyReLU(alpha=0.1)),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print(f"Model built successfully")
    print(f"Total parameters: {model.count_params():,}")

    return model


def train_model(df, sequence_length=7, epochs=50, batch_size=32):
    """Train LSTM model on historical pricing data"""
    try:
        preprocessor = PricePreprocessor()
        df_processed = preprocessor.fit_transform(df)

        X, y = prepare_sequences(df_processed, sequence_length=sequence_length)

        if len(X) == 0:
            print("Not enough data to create sequences!")
            return None, None, None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, shuffle=True
        )

        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        print(f"\nTraining model for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        print(f"Model saved")

        print("TRAINING SUMMARY:")
        print(f"\tTotal Epochs Run: {len(history.history['loss'])}")
        print(f"\tFinal Training Loss: {history.history['loss'][-1]:.6f}")
        print(f"\tFinal Validation Loss: {history.history['val_loss'][-1]:.6f}")
        print(f"\tFinal Training MAE: {history.history['mae'][-1]:.4f}")
        print(f"\tFinal Validation MAE: {history.history['val_mae'][-1]:.4f}")

        return model, preprocessor, (X_val, y_val)

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict_price(df, model, preprocessor, sequence_length=7):
    """Predict price for given input data"""
    try:
        if len(df) == 0:
            raise ValueError("Empty dataframe")

        fallback_price = float(df['current_price'].iloc[-1]) if 'current_price' in df.columns else 0.0

        df_processed = preprocessor.transform(df)

        if len(df_processed) < sequence_length:
            return fallback_price

        feature_cols = [col for col in df_processed.columns
                       if col not in ['current_price', 'dish_id', 'timestamp', 'dt', 'event_name', 'rest_id']]

        X = df_processed[feature_cols].tail(sequence_length).values
        X = X.reshape(1, sequence_length, len(feature_cols))

        predicted_scaled = model.predict(X, verbose=0)[0][0]
        predicted_price = preprocessor.inverse_transform_price([predicted_scaled])[0]

        if predicted_price <= 0 or np.isnan(predicted_price) or np.isinf(predicted_price):
            return fallback_price

        return float(predicted_price)

    except Exception as e:
        print(f"Prediction error: {e}")
        return float(df['current_price'].iloc[-1]) if 'current_price' in df.columns else 0.0


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, preprocessor, X_val, y_val):
    """Evaluate model performance"""

    y_pred_scaled = model.predict(X_val, verbose=0).flatten()

    y_pred = preprocessor.inverse_transform_price(y_pred_scaled)
    y_true = preprocessor.inverse_transform_price(y_val)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    print("MODEL PERFORMANCE METRICS:")
    print(f"\tMean Absolute Error (MAE):     ₹{mae:.2f}")
    print(f"\tRoot Mean Squared Error (RMSE): ₹{rmse:.2f}")
    print(f"\tR² Score:                       {r2:.4f}")
    print(f"\tMean Absolute Percentage Error: {mape:.2f}%")

    accuracy = max(0, 100 - mape)
    print(f"\nModel Accuracy: ~{accuracy:.1f}%")

    print("\nSAMPLE PREDICTIONS (First 10)")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Difference':<15} {'Error %'}")
    print("-" * 70)

    for i in range(min(10, len(y_true))):
        diff = y_pred[i] - y_true[i]
        error_pct = abs(diff / (y_true[i] + 1e-10)) * 100
        print(f"₹{y_true[i]:>8.2f}        ₹{y_pred[i]:>8.2f}          "
              f"{'₹' + f'{diff:>6.2f}':<15} {error_pct:>5.1f}%")

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy
    }

def demonstrate_prediction(df, model, preprocessor):
    """Demonstrate live prediction on sample dishes"""

    sample_dishes = df.groupby('rest_id')['dish_id'].first().values[:3]
    print("\n Predicting prices for sample dishes...\n")

    for i, dish_id in enumerate(sample_dishes, 1):
        dish_data = df[df['dish_id'] == dish_id].sort_values('timestamp')

        if len(dish_data) < 1:
            continue

        current_price = dish_data['current_price'].iloc[-1]
        predicted_price = predict_price(dish_data, model, preprocessor)

        change = predicted_price - current_price
        change_pct = (change / current_price) * 100

        print(f"{i}. Dish ID: {dish_id}")
        print(f"\tRestaurant: {dish_data['rest_id'].iloc[0]}")
        print(f"\tCurrent Price:   ₹{current_price:.2f}")
        print(f"\tPredicted Price: ₹{predicted_price:.2f}")
        print(f"\tChange:          ₹{change:+.2f} ({change_pct:+.1f}%)")

        if change > 0:
            print(f"Recommendation: INCREASE price")
        elif change < 0:
            print(f"Recommendation: DECREASE price")
        else:
            print(f"Recommendation: MAINTAIN price")
        print()

def main():
    # Step 1: Load Data
    if not Path(DATA_FILE).exists():
        print(f"\nError: {DATA_FILE} not found!")
        return

    df = pd.read_csv(DATA_FILE)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(['dish_id', 'timestamp']).reset_index(drop=True)

    print(f"Records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date Range: {df['dt'].min()} to {df['dt'].max()}")
    print(f"Unique Dishes: {df['dish_id'].nunique()}")
    print(f"Unique Restaurants: {df['rest_id'].nunique()}")

    # Step 2: Train Model
    model, preprocessor, val_data = train_model(
        df, sequence_length=SEQUENCE_LENGTH,
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    if model is None or preprocessor is None or val_data is None:
        print("\nTraining failed, exiting...")
        return

    X_val, y_val = val_data

    # Step 3: Evaluate Model
    evaluate_model(model, preprocessor, X_val, y_val)

    # Step 4: Demonstrate Predictions
    demonstrate_prediction(df, model, preprocessor)

    print(f"Model saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()