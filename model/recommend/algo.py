import pandas as pd
from apyori import apriori
from utils.convex import save_data_to_convex

async def cart_recommendation(old_rules:pd.DataFrame,df:pd.DataFrame):
    num_records = len(df)
    if num_records == 0:
        return 
    
    records = []
    for i in range(0,num_records):
        records.append([str(df.value[i,j]) for j in range(0,20)])
    
    association_rules = apriori(records, min_support=0.005, min_confidence=0.2, min_lift=3, min_length=2)
    association_results = list(association_rules)
    for r in association_results:
        lhs = list(r.items)
        rhs = list(r.ordered_statistics[0].items_add)
        support = r.support
        confidence = r.ordered_statistics[0].confidence
        lift = r.ordered_statistics[0].lift

    await save_data_to_convex("rules","insertRule",{lhs,rhs,support,confidence,lift})
    

def history_recommendation(df:pd.DataFrame):
    return