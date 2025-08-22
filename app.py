from fastapi import FastAPI
from controllers import pricing_controller, recommend_controller

app = FastAPI(title="Restaurant Pricing API")

# include pricing router
app.include_router(pricing_controller.router, prefix="/price", tags=["Price"])
app.include_router(recommend_controller.router,prefix="/recommend",tags=["Recommend"])

@app.get("/")
async def getData():
    return {"msg": "Hello"}
