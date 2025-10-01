from datetime import datetime


def to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000)

def is_weekend(dt: datetime) -> bool:
    return True if dt.weekday() >= 5 else False

def time_of_day(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 12: return "MORNING"
    if 12 <= h < 15: return "NOON"
    if 15 <= h < 20: return "AFTERNOON"
    return "NIGHT"

def season(dt: datetime) -> str:
    m = dt.month
    if m in (12, 1, 2): return "WINTER"
    if m in (3, 4, 5, 6): return "SUMMER"
    return "MONSOON"

def day_of_week(dt: datetime) -> str:
    return dt.strftime("%A")

def is_holiday(dt: datetime, event_name: str | None) -> int:
    return 1 if (dt.weekday() == 6 or event_name) else 0