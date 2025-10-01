import httpx
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
CONVEX_URL = os.getenv("CONVEX_URL")
CONVEX_KEY = os.getenv("CONVEX_KEY")

async def fetch_data_from_convex(module:str,function:str):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{CONVEX_URL}/api/query",
            headers = {"Content-Type": "application/json"},
            json = {
            "path": f"{module}:{function}",
            "args": {},
            }
        )
        if res.json().get("error"):
            print("Error fetching data from Convex:", res.json().get("error"))
            return pd.DataFrame()
        else:
            return pd.DataFrame(res.json().get("data"))
     

async def save_data_to_convex(module:str,function:str,args:dict):
    async with httpx.AsyncClient() as client:
        res = client.post(
            f"{CONVEX_URL}/api/mutation",
            headers = {"Content-Type": "application/json"},
            json = {
            "path": f"{module}:{function}",
            "args": args,
            }
        )
        return res.json()