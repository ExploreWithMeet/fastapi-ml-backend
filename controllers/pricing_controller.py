from fastapi import APIRouter
# from schemas.pricing import PricingRequest, PricingResponse
from services.pricing_service import get_price_prediction

router = APIRouter()

# @router.post("/predict", response_model=PricingResponse)
# def predict_price_endpoint(request: PricingRequest):
#     return get_price_prediction(request)

@router.get('/')
async def predict_price():
    return  get_price_prediction()