import numpy as np
import pandas as pd
from joblib import load
from config.paths import PRICING_MODEL_PATH
from schema.pricing import priceRequest, priceResponse
from model.price.preprocessing import preprocess_data

def load_model():
    return load(PRICING_MODEL_PATH)

def predict_price(requests: list[priceRequest]):
    df = pd.DataFrame([r.to_dict_with_features() for r in requests])
    df, _ = preprocess_data(df)

    model = load_model()
    X = df.drop(columns=["predicted_price"], errors="ignore")

    preds = model.predict(X)

    responses = []
    for req, pred in zip(requests, preds):
        responses.append(priceResponse(
            time_of_day=req.time_of_day,
            dish_id=req.dish_id,
            predicted_price=float(pred),
            timestamp=req.timestamp
        ))
    return responses
