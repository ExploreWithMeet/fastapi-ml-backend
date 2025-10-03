import csv
import random
from datetime import datetime, timedelta

# Restaurant configurations with MORE items (20+ items each)
RESTAURANTS = {
    6: list(range(1, 25)),      # Pizza Paradise: 24 items
    13: list(range(25, 45)),    # Burger House: 20 items  
    19: list(range(45, 70)),    # Sushi Station: 25 items
}

# Popular item combinations (for realistic patterns)
COMBOS = {
    6: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11], [12, 13], [14, 15, 16]],
    13: [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35], [36, 37], [38, 39]],
    19: [[45, 46, 47], [48, 49, 50], [51, 52, 53], [54, 55], [56, 57], [58, 59, 60]],
}

def generate_csv():
    """Generate enhanced CSV data"""
    output_file = "fakedata.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['user_id', 'rest_id', 'order_date', 'item_1', 'item_2', 'item_3', 'item_4'])
        
        start_date = datetime(2024, 1, 1)
        
        for rest_id, all_items in RESTAURANTS.items():
            combos = COMBOS[rest_id]
            base_user_id = rest_id * 100
            
            # Generate different types of users
            for user_offset in range(1, 41):  # 40 users per restaurant
                user_id = base_user_id + user_offset
                
                # User patterns
                if user_offset <= 5:
                    # Power users: many orders, love specific combos (will trigger history)
                    num_orders = random.randint(12, 20)
                    favorite_items = random.sample(all_items, 5)  # Has 5 favorite items
                    use_favorites = 0.7
                elif user_offset <= 15:
                    # Regular users: moderate orders, some patterns (history + associations)
                    num_orders = random.randint(6, 12)
                    favorite_items = random.sample(all_items, 8)
                    use_favorites = 0.5
                elif user_offset <= 30:
                    # Casual users: few orders, varied (history + top picks)
                    num_orders = random.randint(3, 6)
                    favorite_items = random.sample(all_items, 10)
                    use_favorites = 0.4
                else:
                    # Light users: very few orders (will need associations + top picks)
                    num_orders = random.randint(1, 3)
                    favorite_items = random.sample(all_items, 12)
                    use_favorites = 0.3
                
                for order_num in range(num_orders):
                    # Order date
                    days_offset = random.randint(0, 270)
                    order_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
                    
                    # Generate order items
                    order_items = []
                    
                    # Sometimes use combos (creates association patterns)
                    if random.random() < 0.4:
                        combo = random.choice(combos)
                        order_items.extend(combo)
                    # Sometimes use favorite items
                    elif random.random() < use_favorites:
                        num_items = random.randint(2, 4)
                        order_items = random.sample(favorite_items, min(num_items, len(favorite_items)))
                    # Sometimes completely random
                    else:
                        num_items = random.randint(2, 4)
                        order_items = random.sample(all_items, num_items)
                    
                    # Ensure 2-4 items per order
                    order_items = order_items[:4]
                    while len(order_items) < 4:
                        order_items.append('')
                    
                    writer.writerow([user_id, rest_id, order_date] + order_items[:4])
    
    print(f"✅ Generated {output_file}")
    print(f"\n📊 Summary:")
    print(f"  Restaurant 6: {len(RESTAURANTS[6])} items")
    print(f"  Restaurant 13: {len(RESTAURANTS[13])} items")
    print(f"  Restaurant 19: {len(RESTAURANTS[19])} items")
    print(f"\n🎯 Test Examples:")
    print(f"  User 605 at Restaurant 6 (power user with history)")
    print(f"  User 1315 at Restaurant 13 (regular user - will show all 3 sources)")
    print(f"  User 1935 at Restaurant 19 (light user - needs associations + top picks)")
    print(f"  User 9999 at Restaurant 6 (new user - only top picks)")

if __name__ == "__main__":
    generate_csv()
    print("\n✅ Done! Replace './fakedata.csv' with './fakedata_enhanced.csv' in your code")