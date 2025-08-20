from fastapi import APIRouter
from fastapi.responses import JSONResponse
from model.price.train import train_and_save
from schema.pricing import priceRequest, priceResponse
from model.price.model import predict_price
import pandas as pd

router = APIRouter()

@router.post("/predict", response_model=priceResponse)
async def predict_price_controller(request: priceRequest):
    """
    Endpoint to predict dish price.
    Returns only: time_of_day, dish_id, predicted_price
    """
    # Convert timestamp
    ts = pd.to_datetime(request.timestamp, unit="ms")

    # Determine time of day
    hour = ts.hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    # Prepare features DataFrame
    row = {
        "dish_id": request.dish_id,
        "sales": 0,   # TODO: hook from DB
        "demand": 1 if request.demand_7d == "HIGH" else 0,
        "timestamp": ts,
    }
    df = pd.DataFrame([row])

    # Call model
    predicted = predict_price(request.dish_id, df)

    return priceResponse(
        time_of_day=time_of_day,
        dish_id=request.dish_id,
        predicted_price=predicted
    )


@router.post("/train")
async def trainModel(req:priceRequest):
    if req.predicted_price == None:
        return train_and_save()
    else:
        return JSONResponse({"message":"Price was already Predicted"},status_code=403)