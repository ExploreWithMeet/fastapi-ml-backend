from datetime import datetime
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from utils.time_conversion import (
    to_datetime, is_weekend, time_of_day,
    season, day_of_week
)
from config.constant import events


class PriceRequest(BaseModel):
    dish_id: Annotated[str, Field(..., description="Dish ID")]
    current_price: Annotated[float, Field(..., description="Current Price")]
    predicted_price: Annotated[Optional[float], Field(None, description="Predicted Price (filled after model runs)")]
    demand_7d: Annotated[str, Field(..., description="Demand in last 7 days (HIGH/MEDIUM/LOW)")]
    rating_7d: Annotated[int, Field(..., ge=1, le=5, description="Rating in last 7 days (1–5)")]
    timestamp: Annotated[float, Field(..., description="Timestamp in milliseconds")]
    event_name: Annotated[Optional[str], Field(None, description="Event name (if any)")]

    @property
    def dt(self) -> datetime:
        """Base datetime object from timestamp"""
        return to_datetime(self.timestamp)

    @property
    def iso_timestamp(self) -> str:
        return self.dt.isoformat()

    @property
    def is_weekend(self) -> bool:
        return is_weekend(self.dt)

    @property
    def time_of_day(self) -> str:
        return time_of_day(self.dt)

    @property
    def season(self) -> str:
        return season(self.dt)

    @property
    def is_event(self) -> bool:
        if self.event_name:
            return self.event_name.upper() in events
        return False

    @property
    def is_holiday(self) -> bool:
        # Holiday if Sunday OR Event
        return self.dt.weekday() == 6 or self.is_event

    @property
    def day_of_week(self) -> int:
        return self.dt.weekday()  # returns 0=Monday ... 6=Sunday


class priceResponse(BaseModel):
    time_of_day: Annotated[Literal["MORNING","NOON","AFTERNOON","NIGHT"],Field(...,description="PREDICTEDS_TIME_OF_DAY")]
    dish_id: Annotated[str,Field(...,description="PREDICTEDS_DISH_ID")]
    predicted_price: Annotated[float,Field(...,description="PREDICTED_PRICE")]