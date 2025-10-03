import pandas as pd
from apyori import apriori
import json
from datetime import datetime

TEMP_DATA = "./fakedata.csv"
TEMP_CART = "./cart_rules.json"

def generate_rules_for_group(df):
    num_records = len(df)
    if num_records < 2:
        return []

    records = []
    item_columns = [col for col in df.columns if col != 'rest_id']
    
    for i in range(num_records):
        transaction = [str(item) for item in df[item_columns].iloc[i].dropna()]
        if transaction:
            records.append(transaction)

    if not records:
        return []

    min_support_val = 2 / len(records) if len(records) > 0 else 0.005
    
    association_rules = apriori(
        records, 
        min_support=min_support_val, 
        min_confidence=0.1, 
        min_lift=1.2, 
        min_length=2
    )
    association_results = list(association_rules)

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


def load_existing_rules():
    try:
        with open(TEMP_CART, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def process_and_save_all_rules():
    df = pd.read_csv(TEMP_DATA)
    
    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)
    
    if 'rest_id' not in df.columns:
        print("Error: rest_id column not found")
        return

    all_rules_data = load_existing_rules()
    rules_dict = {int(item['rest_id']): item for item in all_rules_data}
    
    grouped = df.groupby('rest_id')
    current_date = datetime.now().isoformat()

    for rest_id, group_df in grouped:
        rules = generate_rules_for_group(group_df)

        if rules:
            rules_dict[int(rest_id)] = {
                "rest_id": int(rest_id),
                "rules": rules,
                "Date": current_date
            }

    updated_rules_list = list(rules_dict.values())

    try:
        with open(TEMP_CART, 'w') as f:
            json.dump(updated_rules_list, f, indent=4)
        print(f"Rules saved to {TEMP_CART}")
        print(f"Total restaurants processed: {len(updated_rules_list)}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    process_and_save_all_rules()