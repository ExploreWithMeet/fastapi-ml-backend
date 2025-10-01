from fastapi import APIRouter
from fastapi.responses import JSONResponse
from model.recommend.algo import cart_recommendation, history_recommendation
from utils.convex import fetch_data_from_convex

router = APIRouter()

@router.post("/history_based")
async def history_based():
    history_recommendation()
    return JSONResponse()

async def cart_based():
    old_rules = await fetch_data_from_convex("rules","get_all_rules")
    df = await fetch_data_from_convex("prices","get_all_prices")
    print(df)
    await cart_recommendation(old_rules,df)
    