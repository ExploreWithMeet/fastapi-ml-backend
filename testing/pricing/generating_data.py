"""
generate_pricing_data.py - Generate realistic time-series pricing data for demonstration

Run: python generate_pricing_data.py

This generates a CSV with realistic restaurant pricing data over 2 years
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RESTAURANTS = 5
DISHES_PER_RESTAURANT = 10
START_DATE = datetime.now() - timedelta(days=365)  # 2 years ago
END_DATE = datetime.now()

# Indian restaurant dish categories with base prices
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

# Events in India (approximate dates)
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
    # Base demand by time of day
    if 12 <= hour < 15 or 19 <= hour < 22:  # Lunch and dinner rush
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[60, 30, 10])[0]
    elif 7 <= hour < 12 or 15 <= hour < 19:  # Moderate times
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[20, 50, 30])[0]
    else:  # Off-peak
        base_demand = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[5, 25, 70])[0]
    
    # Boost demand on events and weekends
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
    
    # Demand-based pricing
    if demand == "HIGH":
        price *= random.uniform(1.15, 1.30)  # 15-30% increase
    elif demand == "MEDIUM":
        price *= random.uniform(1.05, 1.15)  # 5-15% increase
    else:  # LOW
        price *= random.uniform(0.85, 0.95)  # 5-15% decrease
    
    # Rating impact
    rating_multiplier = 0.95 + (rating - 3) * 0.05  # Rating 3 = 1.0x, 5 = 1.1x, 1 = 0.9x
    price *= rating_multiplier
    
    # Event premium
    if is_event:
        price *= random.uniform(1.10, 1.25)
    
    # Weekend premium
    if is_weekend:
        price *= random.uniform(1.05, 1.15)
    
    # Long-term trend (inflation/deflation)
    price *= trend_factor
    
    # Add some random noise
    price *= random.uniform(0.98, 1.02)
    
    return round(price, 2)


def generate_data():
    """Generate complete pricing dataset"""
    print("🚀 Starting data generation...")
    
    all_data = []
    
    # Generate dishes
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
    
    print(f"✅ Generated {len(dishes)} dishes across {NUM_RESTAURANTS} restaurants")
    
    # Generate time series data
    current_date = START_DATE
    day_count = 0
    
    while current_date <= END_DATE:
        day_count += 1
        
        # Progress indicator
        if day_count % 100 == 0:
            print(f"📅 Processing day {day_count}/730...")
        
        # Calculate trend factor (slight inflation over time)
        days_elapsed = (current_date - START_DATE).days
        trend_factor = 1 + (days_elapsed / 730) * 0.15  # 15% price increase over 2 years
        
        # Generate data for each dish at different times of day
        for dish in dishes:
            # Generate 2-4 price points per day (different meal times)
            num_entries = random.randint(2, 4)
            
            for _ in range(num_entries):
                # Random hour within restaurant hours (7 AM - 11 PM)
                hour = random.choice([7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22])
                timestamp_dt = current_date.replace(hour=hour, minute=random.randint(0, 59))
                timestamp_ms = int(timestamp_dt.timestamp() * 1000)
                
                # Calculate features
                is_weekend = timestamp_dt.weekday() >= 5
                day_of_week = timestamp_dt.weekday()
                event_name = get_event_name(timestamp_dt)
                is_event = event_name is not None
                is_holiday = day_of_week == 6 or is_event  # Sunday or event
                
                # Rating varies by restaurant and time (with some randomness)
                base_rating = 3 + (dish["rest_id"] % 3)  # Restaurants have different base ratings
                rating = min(5, max(1, base_rating + random.choice([-1, 0, 0, 1])))
                
                # Demand
                demand = get_demand_level(hour, day_of_week, is_event, is_weekend)
                
                # Calculate dynamic price
                current_price = calculate_dynamic_price(
                    dish["base_price"],
                    demand,
                    rating,
                    is_event,
                    is_weekend,
                    trend_factor
                )
                
                # Time of day
                if 5 <= hour < 12:
                    time_of_day = "MORNING"
                elif 12 <= hour < 15:
                    time_of_day = "NOON"
                elif 15 <= hour < 20:
                    time_of_day = "AFTERNOON"
                else:
                    time_of_day = "NIGHT"
                
                # Season
                month = timestamp_dt.month
                if month in [12, 1, 2]:
                    season = "WINTER"
                elif month in [3, 4, 5, 6]:
                    season = "SUMMER"
                else:
                    season = "MONSOON"
                
                # Create record
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
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp
    df = df.sort_values(['dish_id', 'timestamp']).reset_index(drop=True)
    
    print(f"\n✅ Generated {len(df)} total records")
    print(f"📊 Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"🍽️  Restaurants: {NUM_RESTAURANTS}")
    print(f"🥘 Dishes per restaurant: {DISHES_PER_RESTAURANT}")
    
    # Save to CSV
    filename = "pricing_data.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Data saved to {filename}")
    
    # Print sample statistics
    print("\n" + "="*60)
    print("📈 DATASET STATISTICS")
    print("="*60)
    print(f"Total Records: {len(df):,}")
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
    
    # Show sample records
    print("\n" + "="*60)
    print("📋 SAMPLE RECORDS (First 5)")
    print("="*60)
    print(df.head().to_string())
    
    return df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 PRICING DATA GENERATOR")
    print("="*60)
    print("Generating realistic time-series pricing data for ML model...")
    print()
    
    df = generate_data()
    
    print("\n" + "="*60)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 File created: pricing_data.csv")
    print("🚀 Ready for model training!")
    print()