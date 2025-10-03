"""
app.py - FastAPI main application with scheduled tasks
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from asyncio import sleep, create_task, gather
from contextlib import asynccontextmanager

from controllers import pricing_controller, recommend_controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Runs background tasks every 24 hours.
    """
    async def daily_recommendation_update():
        """Update recommendation rules every 24 hours"""
        while True:
            try:
                print("Starting recommendation rule update...")
                result = await recommend_controller.scheduled_rule_update()
                print(f"Recommendation update: {result['message']}")
            except Exception as e:
                print(f"Error in recommendation update: {e}")
            await sleep(24 * 60 * 60)
    
    async def daily_pricing_update():
        """Train model and predict prices every 24 hours"""
        while True:
            try:
                # Step 1: Train the model
                print("Starting pricing model training...")
                train_result = await pricing_controller.scheduled_train_pricing()
                print(f"Pricing model trained: {train_result}")
                
                # Step 2: Predict prices for all dishes
                print("Starting price predictions...")
                predict_result = await pricing_controller.scheduled_predict_prices()
                print(f"Price predictions: {predict_result}")
                
            except Exception as e:
                print(f"Error in pricing update: {e}")
            
            await sleep(24 * 60 * 60)  # Wait 24 hours
    
    # Start both background tasks
    tasks = [
        create_task(daily_recommendation_update()),
        create_task(daily_pricing_update())
    ]
    
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        print("Shutting down background tasks...")


# Create FastAPI app
app = FastAPI(
    title="Restaurant ML Backend API",
    description="Dynamic Pricing & Recommendation System",
    version="2.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(
    pricing_controller.router,
    prefix="/price",
    tags=["Dynamic Pricing"]
)

app.include_router(
    recommend_controller.router,
    prefix="/recommend",
    tags=["Recommendations"]
)


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Restaurant ML Backend API",
            "status": "running",
            "version": "2.0.0",
            "endpoints": {
                "recommendations": "/recommend/history_based/{user_id}?rest_id={rest_id}",
                "pricing": "/price/predict?dish_id={dish_id}&rest_id={rest_id}",
                "train_pricing": "POST /price/train",
                "train_recommendations": "POST /recommend/update_rules"
            }
        },
        status_code=200
    )


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "service": "ML Backend"},
        status_code=200
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)