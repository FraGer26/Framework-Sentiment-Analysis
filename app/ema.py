import pandas as pd
import streamlit as st
import data
import os
import json

# --- 2. Risk Score Logic (Exponential Moving Average) ---
@st.cache_data
def calculate_risk_score(user_df, half_life=15):
    """
    Calculates the daily Risk Score for a user. Uses disk cache.
    """
    user_id = "unknown"
    if not user_df.empty and "Subject ID" in user_df.columns:
        user_id = str(user_df["Subject ID"].iloc[0])
    
    cache_path = os.path.join(data.EMA_CACHE_DIR, f"ema_{user_id}_h{half_life}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                series = pd.Series(cached_data["scores"])
                series.index = pd.to_datetime(cached_data["dates"])
                return series
        except:
            pass

    # Use cached daily aggregation
    daily_scores = data.calculate_daily_risk(user_df)
    
    if daily_scores.empty:
        return pd.Series()
    
    # Convert index to Datetime for proper plotting/rolling
    daily_scores.index = pd.to_datetime(daily_scores.index)
    
    # 2.3 Apply Exponential Decay
    if half_life <= 0:
        alpha = 1.0 # No decay, instant update
    else:
        alpha = 0.5 ** (1 / half_life)
    
    decayed_scores = []
    prev = 0.0 # Initial state
    
    for val in daily_scores:
        current = alpha * prev + (1 - alpha) * val
        decayed_scores.append(current)
        prev = current
        
    series = pd.Series(decayed_scores, index=daily_scores.index)
    
    # Save to disk
    try:
        out_data = {
            "dates": [str(d.date()) for d in series.index],
            "scores": series.tolist()
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4)
    except:
        pass
        
    return series
