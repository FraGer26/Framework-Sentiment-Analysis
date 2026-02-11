# =============================================================================
# MODULO: data.py
# DESCRIZIONE: Gestisce il caricamento, la cache e le trasformazioni base
#              dei dati del dataset. Fornisce costanti di percorso per la cache
#              e funzioni di utilità usate da tutti gli altri moduli.
# =============================================================================

import pandas as pd  # Libreria per manipolazione dati tabulari
import os  # Libreria per operazioni su file e percorsi
import streamlit as st  # Framework per l'interfaccia web
import hashlib  # Libreria per generare hash (non usato ma disponibile)
import json  # Libreria per lettura/scrittura file JSON

# =============================================================================
# CONFIGURAZIONE DIRECTORY DI CACHE
# Definisce i percorsi dove vengono salvati i risultati dei calcoli
# per evitare di ricalcolarli ad ogni avvio dell'applicazione.
# =============================================================================

# Percorso base della cache: cartella "cache/calculation" dentro la directory dell'app
CACHE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "calculation")
# Sottocartella per i dati filtrati per utente
DATA_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "data")
# Sottocartella per i risultati del calcolo EMA (Media Mobile Esponenziale)
EMA_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "ema")
# Sottocartella per i risultati della segmentazione comportamentale
SEGMENT_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "segments")
# Sottocartella per le statistiche globali calcolate con Spark
GLOBAL_CACHE_DIR = os.path.join(CACHE_BASE_DIR, "global")

# Crea tutte le directory di cache se non esistono già
# exist_ok=True evita errori se la directory esiste già
for d in [DATA_CACHE_DIR, EMA_CACHE_DIR, SEGMENT_CACHE_DIR, GLOBAL_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# FUNZIONE: load_data
# Carica i dati dal file CSV e li converte in un DataFrame Pandas.
# =============================================================================
@st.cache_data  # Decoratore Streamlit: salva il risultato in cache per evitare ricaricamenti
def load_data(file_input):
    """
    Carica i dati di output della classificazione.
    
    L'input può essere:
    - Un percorso file (str): carica dal disco
    - Un oggetto UploadedFile di Streamlit: carica dall'upload dell'utente
    
    Colonne attese nel CSV:
    Subject ID, Chunk, Date, Text, Prob_Severe_Depressed, Prob_Moderate_Depressed
    
    Args:
        file_input: Percorso file (str) o oggetto UploadedFile di Streamlit
        
    Returns:
        pd.DataFrame: DataFrame con i dati caricati, o None se il file non esiste
    """
    # Verifica se l'input è un percorso file (stringa)
    if isinstance(file_input, str):
        # Controlla che il file esista sul disco
        if not os.path.exists(file_input):
            return None  # Restituisce None se il file non è trovato
        # Legge il CSV dal percorso specificato
        df = pd.read_csv(file_input)
    else:
        # L'input è un oggetto UploadedFile di Streamlit (caricato dall'utente)
        df = pd.read_csv(file_input)
    
    # Rimuove eventuali spazi bianchi dai nomi delle colonne
    # Esempio: " Date " diventa "Date"
    df.columns = df.columns.str.strip()
    
    # Converte la colonna 'Date' in formato datetime di Pandas
    # errors='coerce' trasforma i valori non validi in NaT (Not a Time)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Restituisce il DataFrame caricato e pulito
    return df


# =============================================================================
# FUNZIONE: get_subject_ids
# Estrae la lista degli utenti unici dal dataset.
# =============================================================================
@st.cache_data  # Cache Streamlit per evitare ricalcoli
def get_subject_ids(df):
    """
    Restituisce un array con gli ID unici degli utenti nel dataset.
    
    Args:
        df (pd.DataFrame): DataFrame con colonna 'Subject ID'
        
    Returns:
        np.array: Array di ID utente unici
    """
    # Usa .unique() di Pandas per ottenere valori distinti della colonna Subject ID
    return df["Subject ID"].unique()


# =============================================================================
# FUNZIONE: get_user_data
# Filtra il dataset per un singolo utente con cache su disco.
# =============================================================================
@st.cache_data  # Cache Streamlit per evitare filtraggi ripetuti
def get_user_data(df, selected_user):
    """
    Restituisce i dati filtrati per un singolo utente.
    Utilizza cache su disco per velocizzare accessi successivi.
    
    Args:
        df (pd.DataFrame): DataFrame completo del dataset
        selected_user: ID dell'utente da filtrare
        
    Returns:
        pd.DataFrame: Sotto-DataFrame con solo i dati dell'utente selezionato
    """
    # Costruisce il percorso del file cache per l'utente specifico
    cache_path = os.path.join(DATA_CACHE_DIR, f"user_{selected_user}.csv")
    
    # Prova a caricare dalla cache se il file esiste
    if os.path.exists(cache_path):
        try:
            # Legge il CSV dalla cache, parsando la colonna Date come datetime
            return pd.read_csv(cache_path, parse_dates=['Date'])
        except:
            pass  # Se la lettura fallisce, ricalcola i dati
    
    # Filtra il DataFrame originale per l'utente selezionato
    # .copy() crea una copia indipendente per evitare SettingWithCopyWarning
    user_data = df[df["Subject ID"] == selected_user].copy()
    
    # Salva il risultato filtrato su disco per usi futuri
    try:
        user_data.to_csv(cache_path, index=False)  # index=False evita di salvare l'indice
    except:
        pass  # Ignora errori di salvataggio (es. permessi)
    
    # Restituisce il DataFrame filtrato per l'utente
    return user_data


# =============================================================================
# FUNZIONE: calculate_daily_risk
# Calcola il punteggio di rischio medio giornaliero per un utente.
# =============================================================================
@st.cache_data  # Cache Streamlit per evitare ricalcoli costosi
def calculate_daily_risk(user_df):
    """
    Calcola il punteggio di rischio medio giornaliero per un utente.
    
    La formula del rischio è: raw_risk = 2 * Prob_Severe + Prob_Moderate
    Il peso doppio alla depressione severa riflette la sua maggiore gravità clinica.
    
    I valori vengono poi mediati per giorno per ottenere un rischio giornaliero.
    
    Args:
        user_df (pd.DataFrame): DataFrame con i dati di un singolo utente
        
    Returns:
        pd.Series: Serie con indice=data e valori=rischio medio giornaliero
    """
    # Determina l'ID dell'utente per creare un nome file cache univoco
    user_id = "unknown"  # Valore predefinito se l'utente non è identificabile
    if not user_df.empty and "Subject ID" in user_df.columns:
        # Prende l'ID dal primo record del DataFrame
        user_id = str(user_df["Subject ID"].iloc[0])
    
    # Costruisce il percorso del file cache per questo utente
    cache_path = os.path.join(DATA_CACHE_DIR, f"daily_risk_{user_id}.json")
    
    # Prova a caricare il rischio giornaliero dalla cache
    if os.path.exists(cache_path):
        try:
            # Legge il file JSON salvato precedentemente
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Converte il dizionario JSON in una Serie Pandas
                series = pd.Series(data)
                # Converte le chiavi (stringhe) in date
                series.index = pd.to_datetime(series.index).date
                return series  # Restituisce la serie dalla cache
        except:
            pass  # Se il caricamento fallisce, ricalcola

    # Crea una copia del DataFrame per non modificare l'originale
    df = user_df.copy()
    
    # Verifica che le colonne di probabilità esistano
    if 'Prob_Severe_Depressed' not in df.columns or 'Prob_Moderate_Depressed' not in df.columns:
        return pd.Series()  # Restituisce serie vuota se mancano le colonne
    
    # Calcola il punteggio di rischio grezzo: peso doppio per la depressione severa
    df["raw_risk"] = 2 * df["Prob_Severe_Depressed"] + df["Prob_Moderate_Depressed"]
    
    # Raggruppa per data, calcola la media del rischio per ogni giorno e ordina cronologicamente
    daily_means = df.groupby(df["Date"].dt.date)["raw_risk"].mean().sort_index()
    
    # Salva i risultati su disco in formato JSON
    try:
        # Converte le date in stringhe e i valori in float per la serializzazione JSON
        out_dict = {str(k): float(v) for k, v in daily_means.to_dict().items()}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=4)  # indent=4 per leggibilità
    except:
        pass  # Ignora errori di salvataggio
    
    # Restituisce la serie con il rischio medio giornaliero
    return daily_means
