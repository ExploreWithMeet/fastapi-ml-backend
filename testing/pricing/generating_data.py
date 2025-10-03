import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

NUM_RESTAURANTS = 5
DISHES_PER_RESTAURANT = 10
START_DATE = datetime.now() - timedelta(days=365)
END_DATE = datetime.now()

DISH_CATEGORIES = {
    "Starters": ["Paneer Tikka", "Veg Spring Roll", "Gobi 65", "Hara Bhara Kabab"],
    "Main Course": ["Paneer Butter Masala", "Dal Makhani", "Veg Biryani", "Chole Bhature"],
    "Breads": ["Butter Naan", "Tandoori Roti", "Garlic Naan"],
    "Chinese": ["Veg Manchurian", "Hakka Noodles", "Fried Rice"],
    "South Indian": ["Masala Dosa", "Idli Sambar", "Uttapam"],
    "Desserts": ["Gulab Jamun", "Rasmalai", "Ice Cream"]
}

BASE_PRICES = {
    "Starters": (150, 250),
    "Main Course": (200, 350),
    "Breads": (30, 80),
    "Chinese": (120, 200),
    "South Indian": (80, 150),
    "Desserts": (60, 120)
}

INDIAN_EVENTS = {
    "DIWALI": [(datetime(2023, 11, 12), datetime(2023, 11, 14)), 
               (datetime(2024, 11, 1), datetime(2024, 11, 3))],
    "HOLI": [(datetime(2023, 3, 8), datetime(2023, 3, 9)), 
             (datetime(2024, 3, 25), datetime(2024, 3, 26))],
    "EID": [(datetime(2023, 4, 22), datetime(2023, 4, 23)), 
            (datetime(2024, 4, 11), datetime(2024, 4, 12))],
    "CHRISTMAS": [(datetime(2023, 12, 25), datetime(2023, 12, 25)), 
                  (datetime(2024, 12, 25), datetime(2024, 12, 25))],
    "NEW_YEAR": [(datetime(2024, 1, 1), datetime(2024, 1, 1)), 
                 (datetime(2025, 1, 1), datetime(2025, 1, 1))]
}


def get_event_name(date):
    """Check if date falls on an event"""
    for event_name, date_ranges in INDIAN_EVENTS.items():
        for start, end in date_ranges:
            if start.date() <= date.date() <= end.date():
                return event_name
    return None


def get_demand_level(hour, day_of_week, is_event, is_weekend):
    """Calculate demand level based on time and context"""
    if 12 <= hour < 15 or 19 <= hour < 22:  
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[60, 30, 10])[0]
    elif 7 <= hour < 12 or 15 <= hour < 19:  # Moderate times
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[20, 50, 30])[0]
    else:  
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[5, 25, 70])[0]
    
    if is_event:
        if base_demand == "LOW":
            base_demand = "MEDIUM"
        elif base_demand == "MEDIUM":
            base_demand = "HIGH"
    
    if is_weekend and base_demand == "LOW":
        base_demand = random.choices(["MEDIUM", "LOW"], weights=[60, 40])[0]
    
    return base_demand


def calculate_dynamic_price(base_price, demand, rating, is_event, is_weekend, trend_factor):
    """Calculate price based on various factors"""
    price = base_price
    
    if demand == "HIGH":
        price *= random.uniform(1.15, 1.30) 
    elif demand == "MEDIUM":
        price *= random.uniform(1.05, 1.15)  
    else:  
        price *= random.uniform(0.85, 0.95) 
    
    rating_multiplier = 0.95 + (rating - 3) * 0.05  
    price *= rating_multiplier
    
    if is_event:
        price *= random.uniform(1.10, 1.25)
    
    if is_weekend:
        price *= random.uniform(1.05, 1.15)
    
    price *= trend_factor
    
    price *= random.uniform(0.98, 1.02)
    
    return round(price, 2)


def generate_data():
    """Generate complete pricing dataset"""
    
    all_data = []
    
    dishes = []
    for rest_id in range(1, NUM_RESTAURANTS + 1):
        dish_count = 0
        for category, dish_list in DISH_CATEGORIES.items():
            for dish_name in dish_list:
                if dish_count >= DISHES_PER_RESTAURANT:
                    break
                
                dish_id = f"DISH_{rest_id}_{dish_count + 1}"
                base_price = random.uniform(*BASE_PRICES[category])
                
                dishes.append({
                    "dish_id": dish_id,
                    "rest_id": rest_id,
                    "dish_name": dish_name,
                    "category": category,
                    "base_price": round(base_price, 2)
                })
                dish_count += 1
            
            if dish_count >= DISHES_PER_RESTAURANT:
                break
    
    print(f"Generated {len(dishes)} dishes across {NUM_RESTAURANTS} restaurants")
    
    current_date = START_DATE
    day_count = 0
    
    while current_date <= END_DATE:
        day_count += 1
        
        if day_count % 100 == 0:
            print(f"Processing day {day_count}/365...")
        
        days_elapsed = (current_date - START_DATE).days
        trend_factor = 1 + (days_elapsed / 730) * 0.15 
        for dish in dishes:
            num_entries = random.randint(2, 4)
            
            for _ in range(num_entries):
                hour = random.choice([7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22])
                timestamp_dt = current_date.replace(hour=hour, minute=random.randint(0, 59))
                timestamp_ms = int(timestamp_dt.timestamp() * 1000)
                
                is_weekend = timestamp_dt.weekday() >= 5
                day_of_week = timestamp_dt.weekday()
                event_name = get_event_name(timestamp_dt)
                is_event = event_name is not None
                is_holiday = day_of_week == 6 or is_event 
                
                base_rating = 3 + (dish["rest_id"] % 3)  
                rating = min(5, max(1, base_rating + random.choice([-1, 0, 0, 1])))
                
                demand = get_demand_level(hour, day_of_week, is_event, is_weekend)
                
                current_price = calculate_dynamic_price(
                    dish["base_price"],
                    demand,
                    rating,
                    is_event,
                    is_weekend,
                    trend_factor
                )
                
                if 5 <= hour < 12:
                    time_of_day = "MORNING"
                elif 12 <= hour < 15:
                    time_of_day = "NOON"
                elif 15 <= hour < 20:
                    time_of_day = "AFTERNOON"
                else:
                    time_of_day = "NIGHT"
                
                month = timestamp_dt.month
                if month in [12, 1, 2]:
                    season = "WINTER"
                elif month in [3, 4, 5, 6]:
                    season = "SUMMER"
                else:
                    season = "MONSOON"
                
                record = {
                    "dish_id": dish["dish_id"],
                    "rest_id": dish["rest_id"],
                    "current_price": current_price,
                    "demand_7d": demand,
                    "rating_7d": rating,
                    "timestamp": timestamp_ms,
                    "event_name": event_name if event_name else "",
                    "is_weekend": int(is_weekend),
                    "time_of_day": time_of_day,
                    "season": season,
                    "day_of_week": day_of_week,
                    "is_event": int(is_event),
                    "is_holiday": int(is_holiday)
                }
                
                all_data.append(record)
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(all_data)
    
    df = df.sort_values(['dish_id', 'timestamp']).reset_index(drop=True)
    
    filename = "pricing_data.csv"
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")
    
    print("STATISTICS:")
    print(f"\nTotal Records: {len(df):,}")
    print(f"Unique Dishes: {df['dish_id'].nunique()}")
    print(f"Unique Restaurants: {df['rest_id'].nunique()}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nPrice Statistics:")
    print(f"  Min Price: ₹{df['current_price'].min():.2f}")
    print(f"  Max Price: ₹{df['current_price'].max():.2f}")
    print(f"  Mean Price: ₹{df['current_price'].mean():.2f}")
    print(f"  Median Price: ₹{df['current_price'].median():.2f}")
    print(f"\nDemand Distribution:")
    print(df['demand_7d'].value_counts())
    print(f"\nRating Distribution:")
    print(df['rating_7d'].value_counts().sort_index())
    
    print("SAMPLE RECORDS (First 5)")
    print(df.head().to_string())
    
    return df


if __name__ == "__main__":
    df = generate_data()
    