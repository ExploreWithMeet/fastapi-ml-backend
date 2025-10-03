#1316 13 1914 19 606 6
import json
import pandas as pd
from collections import Counter
from datetime import datetime

TEMP_DATA = "./fakedata.csv"
TEMP_CART = "./cart_rules.json"
TEMP_HISTORY = "./history_rules.json"


def get_user_order_history(user_id, rest_id, df):
    user_data = df[(df['user_id'] == user_id) & (df['rest_id'] == rest_id)]
    return user_data


def get_items_from_orders(orders_df):
    item_columns = [col for col in orders_df.columns 
                   if col not in ['user_id', 'rest_id', 'order_date']]
    
    item_counter = Counter()
    for _, row in orders_df.iterrows():
        for item in row[item_columns]:
            if pd.notna(item):
                item_counter[str(int(item))] += 1
    
    sorted_items = [item for item, count in item_counter.most_common()]
    return sorted_items, item_counter


def get_associated_items(user_items, rules):
    if not user_items or not rules:
        return []
    
    user_items_set = set(user_items)
    associated_items = {}
    
    for rule in rules:
        rule_base = set()
        if isinstance(rule.get("items_base"), list):
            rule_base = set(str(item) for item in rule["items_base"])
        
        rule_add = []
        if isinstance(rule.get("items_add"), list):
            rule_add = [str(item) for item in rule["items_add"]]
        
        if not rule_base or not rule_add:
            continue
        
        intersection = rule_base.intersection(user_items_set)
        if intersection:
            for item in rule_add:
                if item not in user_items_set:
                    confidence = rule.get('confidence', 0)
                    if item not in associated_items or confidence > associated_items[item]:
                        associated_items[item] = confidence
    
    sorted_items = sorted(associated_items.keys(), 
                         key=lambda x: associated_items[x], 
                         reverse=True)
    
    return sorted_items


def get_top_picks(rest_id, df, exclude_items=None, top_n=10):
    rest_data = df[df['rest_id'] == rest_id]
    exclude_items = set(exclude_items) if exclude_items else set()
    
    item_columns = [col for col in rest_data.columns 
                   if col not in ['user_id', 'rest_id', 'order_date']]
    
    item_counter = Counter()
    for _, row in rest_data.iterrows():
        for item in row[item_columns]:
            if pd.notna(item):
                item_str = str(int(item))
                item_counter[item_str] += 1
    
    filtered_items = [(item, count) for item, count in item_counter.items() 
                      if item not in exclude_items]
    
    top_items = [item for item, count in sorted(filtered_items, 
                                                  key=lambda x: x[1], 
                                                  reverse=True)[:top_n]]
    
    return top_items


def get_user_recommendations(user_id, rest_id, top_n=10):
    df = pd.read_csv(TEMP_DATA)
    
    recommendations = []
    recommendation_breakdown = {
        "from_history": [],
        "from_associations": [],
        "from_top_picks": []
    }
    
    user_orders = get_user_order_history(user_id, rest_id, df)
    based_on_orders = len(user_orders)
    
    if not user_orders.empty:
        user_items, item_counts = get_items_from_orders(user_orders)
        
        recommendations.extend(user_items)
        recommendation_breakdown["from_history"] = user_items.copy()
        
        if len(recommendations) < top_n:
            try:
                with open(TEMP_CART, 'r') as f:
                    all_rules = json.load(f)
                
                restaurant_data = next((data for data in all_rules 
                                       if int(data.get("rest_id")) == rest_id), None)
                
                if restaurant_data and restaurant_data.get("rules"):
                    rules = restaurant_data["rules"]
                    associated_items = get_associated_items(user_items, rules)
                    
                    for item in associated_items:
                        if item not in recommendations and len(recommendations) < top_n:
                            recommendations.append(item)
                            recommendation_breakdown["from_associations"].append(item)
            
            except FileNotFoundError:
                pass
            except json.JSONDecodeError:
                pass
        
        if len(recommendations) < top_n:
            all_top_picks = get_top_picks(rest_id, df, exclude_items=set(), top_n=100)
            
            for item in all_top_picks:
                if len(recommendations) >= top_n:
                    break
                if item not in recommendations:
                    recommendations.append(item)
                    recommendation_breakdown["from_top_picks"].append(item)
        
        recommendation_type = "personalized"
    
    else:
        recommendations = get_top_picks(rest_id, df, top_n=top_n)
        recommendation_breakdown["from_top_picks"] = recommendations.copy()
        recommendation_type = "top_picks"
    
    return {
        "user_id": user_id,
        "rest_id": rest_id,
        "recommendations": recommendations[:top_n],
        "type": recommendation_type,
        "based_on_orders": based_on_orders,
        "breakdown": recommendation_breakdown
    }


def save_user_recommendation(result):
    all_recommendations = []
    
    try:
        with open(TEMP_HISTORY, 'r') as f:
            content = f.read().strip()
            if content:
                all_recommendations = json.loads(content)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    
    result["timestamp"] = datetime.now().isoformat()
    all_recommendations.append(result)
    
    with open(TEMP_HISTORY, 'w') as f:
        json.dump(all_recommendations, f, indent=2)


if __name__ == "__main__":
    user_id = int(input("Enter user_id: "))
    rest_id = int(input("Enter rest_id: "))
    
    result = get_user_recommendations(user_id, rest_id)
    
    print(f"\nRecommendations for User {result['user_id']} at Restaurant {result['rest_id']}")
    print(f"Based on {result['based_on_orders']} previous orders")
    print(f"\tFrom History ({len(result['breakdown']['from_history'])}): {result['breakdown']['from_history']}")
    print(f"\tFrom Associations ({len(result['breakdown']['from_associations'])}): {result['breakdown']['from_associations']}")
    print(f"\tFrom Top Picks ({len(result['breakdown']['from_top_picks'])}): {result['breakdown']['from_top_picks']}")
    print(f"\nFINAL RECOMMENDATIONS ({len(result['recommendations'])}): {result['recommendations']}")
    save_user_recommendation(result)