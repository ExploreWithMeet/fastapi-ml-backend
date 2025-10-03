"""
convex.py - Utility functions for Convex database operations
"""
import httpx
import pandas as pd
from os import getenv
from dotenv import load_dotenv

load_dotenv()
CONVEX_URL = getenv("CONVEX_URL")
CONVEX_KEY = getenv("CONVEX_KEY")


async def fetch_data_from_convex(module, function, args=None, as_dataframe=True):
    if args is None:
        args = {}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CONVEX_URL}/api/query",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {CONVEX_KEY}" if CONVEX_KEY else None
                },
                json={
                    "path": f"{module}:{function}",
                    "args": args
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("error"):
                return pd.DataFrame() if as_dataframe else []
            
            data = result.get("data", [])
            
            if as_dataframe:
                return pd.DataFrame(data) if data else pd.DataFrame()
            else:
                return data
                
    except Exception as e:
        return pd.DataFrame() if as_dataframe else []


async def save_data_to_convex(module, function, args):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{CONVEX_URL}/api/mutation",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {CONVEX_KEY}" if CONVEX_KEY else None
                },
                json={
                    "path": f"{module}:{function}",
                    "args": args
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("error"):
                return {"success": False, "error": result.get("error")}
            
            return {"success": True, "data": result.get("data")}
            
    except Exception as e:
        return {"success": False, "error": str(e)}