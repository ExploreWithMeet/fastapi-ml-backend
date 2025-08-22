from typing import Annotated, List
from pydantic import BaseModel, Field

class recommendRequest(BaseModel):
    dish_ids: Annotated[List[str],Field(...,description=["IDs of Dishes"],examples=["asdasd.sad"])]
    