"""
events.py - Manage event data in Convex
"""
from datetime import datetime, timedelta
from utils.convex import fetch_data_from_convex, save_data_to_convex


async def cleanup_old_events():
    """
    Delete events older than 2 years from Convex
    Called every 24 hours
    """
    try:
        events_df = await fetch_data_from_convex("events", "getAllEvents", as_dataframe=True)
        
        if events_df.empty:
            return {"success": True, "deleted": 0, "message": "No events to clean"}
        
        two_years_ago = datetime.now() - timedelta(days=730)
        two_years_ago_ms = int(two_years_ago.timestamp() * 1000)
        
        old_events = events_df[events_df['timestamp'] < two_years_ago_ms]
        
        deleted_count = 0
        for _, event in old_events.iterrows():
            result = await save_data_to_convex(
                "events",
                "deleteEvent",
                {"eventId": str(event['_id'])}
            )
            if result.get("success"):
                deleted_count += 1
        
        return {
            "success": True,
            "deleted": deleted_count,
            "message": f"Deleted {deleted_count} old events"
        }
        
    except Exception as e:
        return {
            "success": False,
            "deleted": 0,
            "message": str(e)
        }


async def save_event(event_name, timestamp):
    """
    Save a new event to Convex
    
    Args:
        event_name: Name of the event
        timestamp: Timestamp in milliseconds
    """
    try:
        result = await save_data_to_convex(
            "events",
            "createEvent",
            {
                "eventName": event_name,
                "timestamp": timestamp,
                "createdAt": int(datetime.now().timestamp() * 1000)
            }
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def check_if_event_exists(timestamp):
    """
    Check if there's an event on a given timestamp
    
    Args:
        timestamp: Timestamp in milliseconds
        
    Returns:
        dict with event_name or None
    """
    try:
        result = await fetch_data_from_convex(
            "events",
            "getEventByTimestamp",
            args={"timestamp": timestamp},
            as_dataframe=False
        )
        
        if result:
            return {"exists": True, "event_name": result.get("eventName")}
        return {"exists": False, "event_name": None}
        
    except Exception as e:
        return {"exists": False, "event_name": None, "error": str(e)}