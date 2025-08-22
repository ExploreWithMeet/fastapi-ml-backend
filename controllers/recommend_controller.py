from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/history_based")
async def history():
    return JSONResponse

@router.post("/added_based")
async def added_based():
    pass
    return JSONResponse