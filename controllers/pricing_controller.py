import pandas as pd
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from schema.pricing import priceRequest
from utils.convex import fetch_data_from_convex, save_data_to_convex

router = APIRouter()

@router.post("/train")
async def train_model_request(req: List[priceRequest]):
    data = fetch_data_from_convex("prices","get_all_prices")
    df = pd.read_json(data)
    # Convert request list -> DataFrame with enriched features
    # df = pd.DataFrame(item.model_dump() for item in req)
    # save_events(df)
    
    # result = train_model(df)
    # print({
    #     "rows_used": df.shape[0],
    #     "epochs_trained": result["epochs"],
    #     "final_loss": result["final_loss"],
    #     "val_loss": result["val_loss"]
    # })

    return JSONResponse(content={"data":df},status_code=200)


# =======================
# PREDICT ENDPOINT
# =======================
# @router.post("/predict", response_model=priceResponse)
# async def predict_price_endpoint(req: priceRequest):
#     """
#     Predict price for a given request using trained model
#     """
#     # Convert to DataFrame (single record)
#     df = pd.DataFrame([req.to_dict_with_features()])

#     # Load model pipeline
#     predicted_price = predict_price(df)

#     return priceResponse(
#         time_of_day=req.time_of_day,
#         dish_id=req.dish_id,
#         predicted_price=predicted_price,
#         timestamp=req.timestamp
#     )
