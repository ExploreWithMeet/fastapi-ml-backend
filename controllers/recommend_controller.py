"""
recommend_controller.py - FastAPI routes for recommendations
"""
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from model.recommend.algo import (
    generate_all_restaurant_rules,
    get_user_history_recommendations
)

router = APIRouter()


@router.get("/history_based/{user_id}")
async def history_based_recommendations(
    user_id: int,
    rest_id: int = Query(..., description="Restaurant ID"),
    top_n: int = Query(10, description="Number of recommendations")
):
    """
    MAIN API: Get personalized recommendations for a user at a restaurant.
    - If user has order history: Returns AI-powered recommendations
    - If no history: Returns top picks (most popular items)
    
    Path:
        user_id: User ID
    
    Query:
        rest_id: Restaurant ID
        top_n: Number of recommendations (default 10)
    
    Returns:
        JSON: {
            "success": bool,
            "user_id": int,
            "rest_id": int,
            "recommendations": list of item IDs,
            "type": "history" | "top_picks" | "none",
            "message": string,
            "count": int
        }
    """
    try:
        result = await get_user_history_recommendations(
            user_id=user_id,
            rest_id=rest_id,
            top_n=top_n
        )
        
        return JSONResponse(
            content={
                "success": True,
                "user_id": user_id,
                "rest_id": rest_id,
                "recommendations": result["recommendations"],
                "type": result["type"],
                "message": result["message"],
                "count": len(result["recommendations"])
            },
            status_code=200
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": str(e),
                "recommendations": []
            },
            status_code=500
        )


@router.post("/update_rules")
async def manual_update_rules():
    """
    Manually trigger rule generation for all restaurants.
    This is the same function that runs every 24 hours automatically.
    Use for testing or manual updates.
    
    Returns:
        JSON: {
            "success": bool,
            "message": string,
            "restaurants_processed": int,
            "timestamp": string
        }
    """
    try:
        result = await generate_all_restaurant_rules(
        )
        
        return JSONResponse(
            content=result,
            status_code=200 if result["success"] else 500
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": str(e),
                "restaurants_processed": 0
            },
            status_code=500
        )


async def scheduled_rule_update():
    """
    Background task function that runs every 24 hours.
    Called from app.py lifespan function.
    Generates and updates association rules for all restaurants.
    """
    print("Starting scheduled rule update...")
    result = await generate_all_restaurant_rules()
    print(f"Scheduled update complete: {result}")
    return result