import pandas as pd
import os
import streamlit as st
import hashlib
import json

# --- Configurazione Cache ---
CACHE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "calculation")
DATA_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "data")
EMA_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "ema")
SEGMENT_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "segments")
GLOBAL_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "global")

for d in [DATA_CACHE_DIR, EMA_CACHE_DIR, SEGMENT_CACHE_DIR, GLOBAL_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 1. Caricamento Dati ---
@st.cache_data

def load_data(file_input):
    """
    Carica i dati di output della classificazione.
    L'input può essere un percorso file (str) o un oggetto UploadedFile.
    Colonne attese: Subject ID, Chunk, Date, Text, Prob_Severe_Depressed, Prob_Moderate_Depressed
    """
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            return None
        df = pd.read_csv(file_input)
    else:
        # È un oggetto UploadedFile
        df = pd.read_csv(file_input)
    
    # Standardizza nomi colonne se necessario (rimuove spazi)
    df.columns = df.columns.str.strip()
    
    # Converti Data
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df

@st.cache_data
def get_subject_ids(df):
    """Restituisce subject ID unici dal dataframe."""
    return df["Subject ID"].unique()

@st.cache_data
def get_user_data(df, selected_user):
    """Restituisce una copia del dataframe filtrata per ID utente. Usa cache su disco."""
    cache_path = os.path.join(DATA_CACHE_DIR, f"user_{selected_user}.csv")
    
    if os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path, parse_dates=['Date'])
        except:
            pass
            
    user_data = df[df["Subject ID"] == selected_user].copy()
    
    # Salva su disco
    try:
        user_data.to_csv(cache_path, index=False)
    except:
        pass
        
    return user_data

@st.cache_data
def calculate_daily_risk(user_df):
    """Calcola punteggi di rischio medio giornaliero (2*Severe + Moderate). Usa cache su disco."""
    # Crea un ID unico per i dati utente per cache risultato
    # Usiamo user_id se lo abbiamo, ma qui abbiamo solo il DF.
    # Assumiamo che user_df abbia una colonna 'Subject ID'.
    user_id = "unknown"
    if not user_df.empty and "Subject ID" in user_df.columns:
        user_id = str(user_df["Subject ID"].iloc[0])
    
    cache_path = os.path.join(DATA_CACHE_DIR, f"daily_risk_{user_id}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                series = pd.Series(data)
                series.index = pd.to_datetime(series.index).date
                return series
        except:
            pass

    df = user_df.copy()
    if 'Prob_Severe_Depressed' not in df.columns or 'Prob_Moderate_Depressed' not in df.columns:
        return pd.Series()
    
    df["raw_risk"] = 2 * df["Prob_Severe_Depressed"] + df["Prob_Moderate_Depressed"]
    daily_means = df.groupby(df["Date"].dt.date)["raw_risk"].mean().sort_index()
    
    # Salva su disco
    try:
        # Salva come JSON (indice come stringhe)
        out_dict = {str(k): float(v) for k, v in daily_means.to_dict().items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=4)
    except:
        pass
        
    return daily_means
