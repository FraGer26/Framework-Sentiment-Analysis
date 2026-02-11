# =============================================================================
# MODULO: ema.py
# DESCRIZIONE: Implementa il calcolo del punteggio di rischio usando la
#              Media Mobile Esponenziale (EMA - Exponential Moving Average).
#              L'EMA dà più peso ai valori recenti, permettendo di rilevare
#              tendenze recenti nel rischio di depressione di un utente.
# =============================================================================

import pandas as pd  # Libreria per manipolazione dati tabulari
import streamlit as st  # Framework per interfaccia web (usato per cache)
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from p0_global import data  # Modulo locale per costanti e funzioni dati


def compute_risk_series(user_df, half_life=15, smoothing=True):
    """
    Calcola la serie temporale del punteggio di rischio giornaliero per un utente.
    
    L'algoritmo funziona in 3 fasi:
    1. Aggrega il rischio grezzo per giorno (media giornaliera)
    2. Applica decadimento esponenziale (EMA) con il half-life specificato
    3. Opzionalmente applica media mobile centrata a 7 giorni per smoothing
    
    Il decadimento esponenziale fa sì che i valori recenti abbiano più peso:
    - half_life=15 significa che dopo 15 giorni un valore pesa il 50%
    - half_life=0 significa aggiornamento istantaneo (nessun decadimento)
    
    La formula EMA è: current = alpha * previous + (1 - alpha) * value
    dove alpha = 0.5^(1/half_life)
    
    Questa funzione è separata dalla logica di caching per poter essere
    riutilizzata anche nei contesti Spark (es. compute_risk_rankings).
    
    Args:
        user_df (pd.DataFrame): DataFrame con i dati di un singolo utente
        half_life (int): Periodo di dimezzamento in giorni (default: 15)
        smoothing (bool): Se True, applica media mobile a 7 giorni (default: True)
        
    Returns:
        pd.Series: Serie con indice=data e valori=punteggio di rischio EMA
    """
    # Fase 1: Calcola il rischio grezzo medio per ogni giorno
    # Usa la funzione di data.py che applica formula: 2*Severe + Moderate
    daily_scores = data.calculate_daily_risk(user_df)
    
    # Se non ci sono dati sufficienti, restituisce serie vuota
    if daily_scores.empty:
        return pd.Series()
    
    # Converte l'indice da date a datetime per compatibilità con rolling e plotting
    daily_scores.index = pd.to_datetime(daily_scores.index)
    
    # Fase 2: Calcola il fattore di decadimento alpha
    if half_life <= 0:
        # Se half_life è 0 o negativo, nessun decadimento: il valore corrente è l'unico peso
        alpha = 1.0
    else:
        # Calcola alpha dalla formula del dimezzamento
        # alpha = 0.5^(1/15) ≈ 0.955 per half_life=15
        # Significa che ogni giorno il valore precedente pesa circa il 95.5%
        alpha = 0.5 ** (1 / half_life)
    
    # Fase 2: Applica il decadimento esponenziale iterativamente
    decayed_scores = []  # Lista per accumulare i valori EMA
    prev = 0.0  # Stato iniziale del filtro EMA
    
    for val in daily_scores:  # Itera su ogni valore giornaliero
        # Formula EMA: combina il valore precedente (pesato alpha) con il nuovo (pesato 1-alpha)
        current = alpha * prev + (1 - alpha) * val
        decayed_scores.append(current)  # Aggiunge il valore EMA alla lista
        prev = current  # Aggiorna lo stato per la prossima iterazione
    
    # Converte la lista in una Serie Pandas con le stesse date come indice
    series = pd.Series(decayed_scores, index=daily_scores.index)
    
    # Fase 3: Applica media mobile centrata a 7 giorni per lisciare la curva
    if smoothing:
        # window=7: finestra di 7 giorni
        # center=True: la media è centrata (3 giorni prima + giorno corrente + 3 giorni dopo)
        # min_periods=1: calcola anche se ci sono meno di 7 valori nella finestra
        series = series.rolling(window=7, center=True, min_periods=1).mean()
    
    # Restituisce la serie EMA (con o senza smoothing)
    return series


@st.cache_data  # Decoratore Streamlit: salva il risultato in cache per evitare ricalcoli
def calculate_risk_score(user_df, half_life=15):
    """
    Versione con caching del calcolo del rischio EMA.
    
    Salva e carica i risultati da file JSON su disco.
    Usa internamente compute_risk_series con smoothing abilitato.
    
    Args:
        user_df (pd.DataFrame): DataFrame con i dati dell'utente
        half_life (int): Periodo di dimezzamento EMA in giorni (default: 15)
        
    Returns:
        pd.Series: Serie EMA con smoothing, indicizzata per data
    """
    # Determina l'ID utente per il nome del file cache
    user_id = "unknown"  # Valore predefinito
    if not user_df.empty and "Subject ID" in user_df.columns:
        user_id = str(user_df["Subject ID"].iloc[0])  # Prende l'ID dal primo record
    
    # Costruisce il percorso del file cache per questa combinazione utente/half_life
    cache_path = os.path.join(data.EMA_CACHE_DIR, f"ema_{user_id}_h{half_life}_smoothed.json")
    
    # Prova a caricare il risultato dalla cache su disco
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)  # Legge il file JSON
                series = pd.Series(cached_data["scores"])  # Crea Serie dai punteggi
                series.index = pd.to_datetime(cached_data["dates"])  # Ripristina le date come indice
                return series  # Restituisce dalla cache
        except:
            pass  # Se il caricamento fallisce, ricalcola

    # Calcola la serie EMA con smoothing abilitato
    series = compute_risk_series(user_df, half_life=half_life, smoothing=True)
    
    # Se la serie è vuota, la restituisce senza salvare
    if series.empty:
        return series
    
    # Salva i risultati su disco per usi futuri
    try:
        # Prepara i dati per la serializzazione JSON
        out_data = {
            "dates": [str(d.date()) for d in series.index],  # Converte date in stringhe
            "scores": series.tolist()  # Converte valori in lista Python
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4)  # Salva con indentazione per leggibilità
    except:
        pass  # Ignora errori di salvataggio
    
    # Restituisce la serie EMA calcolata
    return series
