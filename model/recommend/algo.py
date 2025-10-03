"""
algo.py - Core recommendation algorithms using Apriori
"""
import pandas as pd
from apyori import apriori
from datetime import datetime
from collections import Counter
from utils.convex import fetch_data_from_convex, save_data_to_convex

def generate_rules_for_restaurant(df):
    num_records = len(df)
    if num_records < 2:
        return []

    records = []
    
    for i in range(num_records):
        transaction = [str(item) for item in df.iloc[i].dropna() if str(item) != 'nan']
        if transaction:
            records.append(transaction)

    if not records:
        return []

    min_support_val = max(2 / len(records), 0.005) if len(records) > 0 else 0.005
    
    try:
        association_rules = apriori(
            records, 
            min_support=min_support_val, 
            min_confidence=0.1, 
            min_lift=1.2, 
            min_length=2
        )
        association_results = list(association_rules)
    except:
        return []

    results = []
    for r in association_results:
        for ordered_statistic in r.ordered_statistics:
            rule = {
                "items_base": list(ordered_statistic.items_base),
                "items_add": list(ordered_statistic.items_add),
                "confidence": ordered_statistic.confidence,
                "lift": ordered_statistic.lift,
                "support": r.support
            }
            results.append(rule)
    
    results.sort(key=lambda x: (x['confidence'], x['support'], x['lift']), reverse=True)
    return results


async def generate_all_restaurant_rules():
    try:
        orders_df = await fetch_data_from_convex("orders", "getAllOrderHistory")
        
        if orders_df.empty:
            return {
                "success": False,
                "message": "No order data found",
                "restaurants_processed": 0
            }
        
        if 'rest_id' not in orders_df.columns and 'restaurant_id' in orders_df.columns:
            orders_df.rename(columns={'restaurant_id': 'rest_id'}, inplace=True)
        
        grouped = orders_df.groupby('rest_id')
        processed_count = 0
        current_date = datetime.now().isoformat()
        
        for rest_id, group_df in grouped:
            transaction_df = group_df.drop(columns=['user_id', 'rest_id'], errors='ignore')
            rules = generate_rules_for_restaurant(transaction_df)
            
            if rules:
                await save_data_to_convex(
                    "rules", 
                    "upsertRestaurantRules",
                    {
                        "restId": int(rest_id),
                        "rules": rules,
                        "date": current_date,
                        "ruleCount": len(rules)
                    }
                )
                processed_count += 1
        
        return {
            "success": True,
            "message": f"Successfully processed {processed_count} restaurants",
            "restaurants_processed": processed_count,
            "timestamp": current_date
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "restaurants_processed": 0
        }


def get_recommendations_from_rules(user_items, rules, top_n=10):
    if not user_items or not rules:
        return []
    
    recommendations = {}
    user_items_str = set(map(str, user_items))

    for rule in rules:
        if set(rule["items_base"]).issubset(user_items_str):
            for item_to_add in rule["items_add"]:
                if item_to_add not in user_items_str:
                    if item_to_add not in recommendations or \
                       rule['confidence'] > recommendations[item_to_add]:
                        recommendations[item_to_add] = rule['confidence']

    sorted_recs = sorted(
        recommendations.keys(), 
        key=lambda item: recommendations[item], 
        reverse=True
    )

    return sorted_recs[:top_n]


async def get_user_history_recommendations(user_id, rest_id,top_n=10):
    try:
        user_orders_df = await fetch_data_from_convex(
            "orders",
            "getUserRestaurantOrders",
            args={"userId": user_id, "restId": rest_id}
        )
        
        if not user_orders_df.empty and len(user_orders_df) > 0:
            user_items = []
            for _, row in user_orders_df.iterrows():
                items = [str(item) for item in row.dropna() 
                        if str(item) not in ['nan', str(user_id), str(rest_id)]]
                user_items.extend(items)
            
            user_items = list(set(user_items))
            
            rules_df = await fetch_data_from_convex(
                "rules",
                "getRestaurantRules",
                args={"restId": rest_id}
            )
            
            if not rules_df.empty and 'rules' in rules_df.columns:
                rules = rules_df['rules'].iloc[0]
                recommendations = get_recommendations_from_rules(user_items, rules, top_n)
                
                if recommendations:
                    return {
                        "recommendations": recommendations,
                        "type": "history",
                        "message": f"Based on your {len(user_orders_df)} previous orders"
                    }
        
        all_orders_df = await fetch_data_from_convex(
            "orders",
            "getRestaurantOrders",
            args={"restId": rest_id}
        )
        
        if all_orders_df.empty:
            return {
                "recommendations": [],
                "type": "none",
                "message": "No data available for this restaurant"
            }
        
        item_counter = Counter()
        for _, row in all_orders_df.iterrows():
            items = [str(item) for item in row.dropna() 
                    if str(item) not in ['nan', 'user_id', 'rest_id']]
            item_counter.update(items)
        
        top_items = [item for item, count in item_counter.most_common(top_n)]
        
        return {
            "recommendations": top_items,
            "type": "top_picks",
            "message": "Popular items at this restaurant"
        }
        
    except Exception as e:
        return {
            "recommendations": [],
            "type": "error",
            "message": str(e)
        }