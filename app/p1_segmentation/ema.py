import pandas as pd
import streamlit as st
from p0_global import data
import os
import json

# --- 2. Logica Punteggio di Rischio (Media Mobile Esponenziale) ---

def compute_risk_series(user_df, half_life=15, smoothing=True):
    """
    Logica principale per calcolare il Punteggio di Rischio giornaliero per un utente.
    Separata dalla logica di caching/Streamlit per riutilizzo in Spark UDFs.
    
    Args:
        user_df (pd.DataFrame): Dati utente
        half_life (int): Half life per EMA
        smoothing (bool): Se applicare media mobile centrata a 7 giorni
        
    Returns:
        pd.Series: Serie punteggio rischio indicizzata per Data
    """
    # Usa aggregazione giornaliera in cache
    daily_scores = data.calculate_daily_risk(user_df)
    
    if daily_scores.empty:
        return pd.Series()
    
    # Converti indice in Datetime per plotting/rolling corretti
    daily_scores.index = pd.to_datetime(daily_scores.index)
    
    # 2.3 Applica Decadimento Esponenziale
    if half_life <= 0:
        alpha = 1.0 # Nessun decadimento, aggiornamento istantaneo
    else:
        alpha = 0.5 ** (1 / half_life)
    
    decayed_scores = []
    prev = 0.0 # Stato iniziale
    
    for val in daily_scores:
        current = alpha * prev + (1 - alpha) * val
        decayed_scores.append(current)
        prev = current
        
    series = pd.Series(decayed_scores, index=daily_scores.index)
    
    # --- Applica Media Mobile Centrata a 7 Giorni (Smoothing) ---
    if smoothing:
        series = series.rolling(window=7, center=True, min_periods=1).mean()
        
    return series

@st.cache_data
def calculate_risk_score(user_df, half_life=15):
    """
    Calcola il Punteggio di Rischio giornaliero per un utente con caching.
    Usa internamente compute_risk_series (con smoothing abilitato di default).
    """
    user_id = "unknown"
    if not user_df.empty and "Subject ID" in user_df.columns:
        user_id = str(user_df["Subject ID"].iloc[0])
    
    cache_path = os.path.join(data.EMA_CACHE_DIR, f"ema_{user_id}_h{half_life}_smoothed.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                series = pd.Series(cached_data["scores"])
                series.index = pd.to_datetime(cached_data["dates"])
                return series
        except:
            pass

    # Chiama la funzione logica
    series = compute_risk_series(user_df, half_life=half_life, smoothing=True)
    
    if series.empty:
        return series
    
    # Salva su disco
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
