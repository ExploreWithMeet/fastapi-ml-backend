from datetime import datetime
from typing import Annotated, Optional
from pydantic import BaseModel, Field, computed_field
from utils.time_conversion import to_datetime, is_weekend, time_of_day, season
from config.constant import events


class priceRequest(BaseModel):
    dish_id: Annotated[str, Field(..., description="Dish ID")]
    rest_id: Annotated[int, Field(..., description="Restaurant ID")]  # ✅ ADDED
    current_price: Annotated[float, Field(..., description="Current Price")]
    predicted_price: Annotated[Optional[float], Field(None, description="Predicted Price")]
    demand_7d: Annotated[str, Field(..., description="Demand in last 7 days (HIGH/MEDIUM/LOW)")]
    rating_7d: Annotated[int, Field(..., ge=1, le=5, description="Rating in last 7 days (1–5)")]
    timestamp: Annotated[float, Field(..., description="Timestamp in milliseconds")]
    event_name: Annotated[Optional[str], Field(None, description="Event name (if any)")]
    # base_price: Annotated[Optional[float], Field(None, description="Base Price")]  # ✅ OPTIONAL if you want it

    @computed_field
    @property
    def dt(self) -> datetime:
        return to_datetime(self.timestamp)

    @computed_field
    @property
    def iso_timestamp(self) -> str:
        return self.dt.isoformat()

    @computed_field
    @property
    def is_weekend(self) -> bool:
        return is_weekend(self.dt)

    @computed_field
    @property
    def time_of_day(self) -> str:
        return time_of_day(self.dt)

    @computed_field
    @property
    def season(self) -> str:
        return season(self.dt)

    @computed_field
    @property
    def is_event(self) -> bool:
        if self.event_name:
            return self.event_name.upper() in events
        return False

    @computed_field
    @property
    def is_holiday(self) -> bool:
        return self.dt.weekday() == 6 or self.is_event

    @computed_field
    @property
    def day_of_week(self) -> int:
        return self.dt.weekday()  # ✅ KEEP THIS - returns 0-6