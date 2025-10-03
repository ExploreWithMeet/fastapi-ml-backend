"""
pricing_controller.py - FastAPI routes for dynamic pricing
"""
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
from utils.convex import fetch_data_from_convex, save_data_to_convex
from model.pricing.train import train_model
from model.pricing.model import predict_price, predict_batch_prices
from model.pricing.events import cleanup_old_events


router = APIRouter()

async def train_pricing_model():
    """
    Train LSTM model on historical pricing data.
    Called automatically every 24 hours.
    """
    try:
        # Fetch all pricing history
        prices_df = await fetch_data_from_convex("prices", "getAllPrices", as_dataframe=True)
        
        if prices_df.empty:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "No pricing data found",
                    "epochs": 0
                },
                status_code=404
            )
        
        # Train model
        result = train_model(prices_df, sequence_length=7, epochs=50, batch_size=32)
        
        # Cleanup old events
        cleanup_result = await cleanup_old_events()
        result["events_cleanup"] = cleanup_result
        
        return JSONResponse(
            content=result,
            status_code=200 if result["success"] else 500
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": str(e),
                "epochs": 0
            },
            status_code=500
        )


@router.post("/predict")
async def predict_dish_price(dish_id: str, rest_id: int):
    """
    Predict optimal price for a specific dish
    
    Args:
        dish_id: Dish ID
        rest_id: Restaurant ID
        
    Returns:
        Predicted price
    """
    try:
        # Fetch historical data for this dish
        dish_history = await fetch_data_from_convex(
            "prices",
            "getDishHistory",
            args={"dishId": dish_id, "restId": rest_id},
            as_dataframe=True
        )
        
        if dish_history.empty:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "No history found for this dish",
                    "predicted_price": None
                },
                status_code=404
            )
        
        # Predict price
        predicted_price = predict_price(dish_history)
        
        return JSONResponse(
            content={
                "success": True,
                "dish_id": dish_id,
                "rest_id": rest_id,
                "predicted_price": round(predicted_price, 2),
                "current_price": float(dish_history['current_price'].iloc[-1])
            },
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": str(e),
                "predicted_price": None
            },
            status_code=500
        )


@router.post("/predict_batch")
async def predict_restaurant_prices(rest_id: int):
    """
    Predict prices for all dishes in a restaurant
    
    Args:
        rest_id: Restaurant ID
        
    Returns:
        Dictionary of predicted prices per dish
    """
    try:
        # Fetch all dishes for restaurant
        restaurant_data = await fetch_data_from_convex(
            "prices",
            "getRestaurantPrices",
            args={"restId": rest_id},
            as_dataframe=True
        )
        
        if restaurant_data.empty:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "No data found for this restaurant",
                    "predictions": {}
                },
                status_code=404
            )
        
        # Predict for all dishes
        predictions = predict_batch_prices(restaurant_data)
        
        return JSONResponse(
            content={
                "success": True,
                "rest_id": rest_id,
                "predictions": predictions,
                "count": len(predictions)
            },
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": str(e),
                "predictions": {}
            },
            status_code=500
        )


async def scheduled_train_pricing():
    """
    Background task function called every 24 hours from app.py
    """
    result = await train_pricing_model()
    return result.body.decode() if hasattr(result, 'body') else result

async def predict_all_prices():
    """
    Predict prices for all dishes across all restaurants
    Called automatically every 24 hours AFTER training
    """
    try:
        # Fetch all unique restaurants
        restaurants_df = await fetch_data_from_convex(
            "prices", 
            "getAllRestaurants",  # You need this Convex function
            as_dataframe=True
        )
        
        if restaurants_df.empty:
            return {
                "success": False,
                "message": "No restaurants found",
                "predictions": 0
            }
        
        total_predictions = 0
        failed = 0
        
        # Predict for each restaurant
        for _, restaurant in restaurants_df.iterrows():
            rest_id = restaurant['rest_id']
            
            try:
                # Get all dishes for this restaurant
                restaurant_data = await fetch_data_from_convex(
                    "prices",
                    "getRestaurantPrices",
                    args={"restId": rest_id},
                    as_dataframe=True
                )
                
                if restaurant_data.empty:
                    continue
                
                # Predict prices
                predictions = predict_batch_prices(restaurant_data)
                
                # Save each prediction to Convex
                for dish_id, predicted_price in predictions.items():
                    save_result = await save_data_to_convex(
                        "prices",
                        "updatePredictedPrice",  # You need this Convex mutation
                        {
                            "dishId": dish_id,
                            "restId": rest_id,
                            "predictedPrice": round(predicted_price, 2),
                            "timestamp": int(datetime.now().timestamp() * 1000)
                        }
                    )
                    
                    if save_result.get("success"):
                        total_predictions += 1
                    else:
                        failed += 1
                        
            except Exception as e:
                print(f"Error predicting for restaurant {rest_id}: {e}")
                failed += 1
                continue
        
        return {
            "success": True,
            "message": f"Predicted prices for {total_predictions} dishes",
            "total_predictions": total_predictions,
            "failed": failed
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "predictions": 0
        }


async def scheduled_predict_prices():
    """
    Background task function called every 24 hours from app.py
    """
    result = await predict_all_prices()
    return result