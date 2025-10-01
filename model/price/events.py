from datetime import datetime, timedelta
import pandas as pd
from config.paths import EVENTS_PATH

def save_events(df:pd.DataFrame):
    df = df[df["is_event"] == True] 
    df.to_csv(EVENTS_PATH,index=False)
    delete_events()
        
def delete_events():
    df = pd.read_csv(EVENTS_PATH)
    cutoff_date = datetime.now() - timedelta(days=2*365)  
    cutoff_timestamp = cutoff_date.timestamp() * 1000
    new_df = df[df["timestamp"] >= cutoff_timestamp]
    new_df.to_csv(EVENTS_PATH,index=False)