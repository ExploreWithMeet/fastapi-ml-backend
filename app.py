from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from controllers import pricing_controller, recommend_controller
from controllers.recommend_controller import cart_based

app = FastAPI(title="Restaurant Pricing API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def daily_task():
        while True:
            try:
                print("Running cart_based task...")
                await cart_based()  # must be async
                print("Task finished, waiting 24h...")
            except Exception as e:
                print("Error in cart_based:", e)
            await asyncio.sleep(24 * 60 * 60)  # 24 hours

    task = asyncio.create_task(daily_task())
    try:
        yield  # FastAPI app runs here
    finally:
        task.cancel()  # cancel background task on shutdown

app = FastAPI(title="Restaurant Pricing API", lifespan=lifespan)

# --------------------------
# Include routers normally
# --------------------------
app.include_router(pricing_controller.router, prefix="/price", tags=["Price"])
app.include_router(recommend_controller.router, prefix="/recommend", tags=["Recommend"])

# --------------------------
# Root endpoint
# --------------------------
@app.get("/")
async def root():
    return {"message": "FASTAPI ML BACKEND FOR RESTAURANT SERVICE."}
