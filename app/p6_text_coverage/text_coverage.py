# =============================================================================
# MODULO: text_coverage.py
# DESCRIZIONE: Calcola la copertura testuale (Text Coverage) confrontando
#              i post grezzi dell'utente con il testo della narrativa generata.
#              A differenza della Topic Coverage, qui si confrontano direttamente
#              i testi (post vs narrativa), non i topic estratti.
# =============================================================================

import numpy as np  # Libreria per calcoli numerici e array
import pandas as pd  # Libreria per manipolazione dati tabulari
from sklearn.metrics.pairwise import cosine_similarity  # Funzione per similarità coseno
import streamlit as st  # Framework per interfaccia web
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
import hashlib  # Libreria per generare hash (disponibile)
from p6_text_coverage.embedding_utils import get_model, embed_texts  # Funzioni per embedding

# Directory dove vengono salvati i risultati della copertura in cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "coverage")


def get_cache_path(user_id, source_type, threshold, model_name):
    """
    Costruisce il percorso del file cache per la copertura testuale.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente ("base" o "traj")
        threshold (float): Soglia di similarità usata
        model_name (str): Nome del modello di embedding
        
    Returns:
        str: Percorso assoluto del file cache
    """
    # Costruisce un nome file univoco con tutti i parametri
    filename = f"text_cov_{user_id}_{source_type}_{threshold}_{model_name.replace('/', '_')}.json"
    return os.path.join(CACHE_DIR, filename)


def check_cache(user_id, source_type, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    """
    Verifica se i risultati della copertura testuale sono in cache.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente ("base" o "traj")
        threshold (float): Soglia di similarità (default: 0.5)
        model_name (str): Nome del modello di embedding
        
    Returns:
        bool: True se il file cache esiste
    """
    path = get_cache_path(user_id, source_type, threshold, model_name)
    return os.path.exists(path)


def load_cache(user_id, source_type, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    """
    Carica i risultati della copertura testuale dalla cache.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente
        threshold (float): Soglia di similarità
        model_name (str): Nome del modello di embedding
        
    Returns:
        tuple: (metriche, df_corrispondenze, None)
    """
    path = get_cache_path(user_id, source_type, threshold, model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Carica il JSON dalla cache
        # Ricostruisce il DataFrame dalla lista di dizionari
        df_matches = pd.DataFrame(data["matches_data"])
        return data["metrics"], df_matches, None
    return None, None, None


def save_cache(user_id, source_type, threshold, model_name, metrics, df_matches):
    """
    Salva i risultati della copertura testuale nella cache su disco.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente
        threshold (float): Soglia di similarità
        model_name (str): Nome del modello di embedding
        metrics (dict): Metriche calcolate
        df_matches (pd.DataFrame): DataFrame con le corrispondenze
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = get_cache_path(user_id, source_type, threshold, model_name)
    # Prepara i dati per la serializzazione
    data = {
        "metrics": metrics,
        "matches_data": df_matches.to_dict(orient="records")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


@st.cache_data  # Cache Streamlit per evitare ricalcoli
def calculate_text_coverage_metrics(user_id, source_type, texts_full, text_sample, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    """
    Calcola le metriche di copertura testuale (Precision, Recall, F1).
    
    Confronta direttamente i post grezzi (texts_full) con il testo
    della narrativa (text_sample) usando embedding e similarità coseno.
    
    A differenza di Topic Coverage, qui:
    - texts_full = lista di post individuali dell'utente
    - text_sample = testo della narrativa (stringa singola o lista)
    
    La soglia predefinita è più bassa (0.5 vs 0.75) perché il confronto
    testo-a-testo è intrinsecamente meno preciso del confronto topic-a-topic.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente ("base" o "traj")
        texts_full (list): Lista dei post grezzi dell'utente
        text_sample: Narrativa (stringa singola o lista di stringhe)
        threshold (float): Soglia minima di similarità (default: 0.5)
        model_name (str): Nome del modello di embedding
        
    Returns:
        tuple: (metriche, df_corrispondenze, matrice_similarità)
    """
    # Verifica che ci siano dati sufficienti
    if not texts_full or not text_sample:
        return None, None, None

    # Carica il modello di embedding
    model = get_model(model_name)
    
    # Genera gli embedding per i testi completi (post grezzi)
    emb_full = embed_texts(model, texts_full)
    
    # Gestisce il formato del testo campione (può essere stringa o lista)
    if isinstance(text_sample, str):
        texts_sample = [text_sample]  # Converte stringa in lista di un elemento
    elif isinstance(text_sample, list):
        texts_sample = text_sample  # Già una lista
    else:
        return None, None, None  # Formato non supportato
    
    # Genera gli embedding per il testo campione
    emb_sample = embed_texts(model, texts_sample)
    
    # Calcola la matrice di similarità coseno (n_full x n_sample)
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # ==========================================================================
    # CALCOLO CORRISPONDENZE
    # Per ogni post, verifica se c'è almeno un componente della narrativa
    # con similarità sufficiente (>= threshold)
    # ==========================================================================
    matches_full = (sim_matrix.max(axis=1) >= threshold)  # Post coperti dalla narrativa
    matches_sample = (sim_matrix.max(axis=0) >= threshold)  # Componenti narrativa che coprono
    
    # Conta i veri positivi
    tp_recall = matches_full.sum()  # Post coperti
    tp_precision = matches_sample.sum()  # Componenti narrativa rilevanti
    
    # Calcola le metriche
    recall = tp_recall / len(texts_full) if texts_full else 0.0
    precision = tp_precision / len(texts_sample) if texts_sample else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Assembla il dizionario delle metriche
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp_recall": int(tp_recall),  # Numero post coperti
        "tp_precision": int(tp_precision),
        "n_full": len(texts_full),  # Numero totale post
        "n_sample": len(texts_sample)
    }
    
    # ==========================================================================
    # DETTAGLI CORRISPONDENZE
    # Per ogni post, trova il componente della narrativa più simile
    # ==========================================================================
    matches_data = []
    for i, t_full in enumerate(texts_full):
        j_best = sim_matrix[i].argmax()  # Indice del componente più simile
        sim = sim_matrix[i, j_best]  # Similarità massima
        t_sample = texts_sample[j_best]  # Testo del componente più simile
        
        matches_data.append({
            "full_text": t_full,  # Testo del post originale
            # Tronca il testo campione a 100 caratteri per la visualizzazione
            "sample_component": t_sample[:100] + "..." if len(t_sample) > 100 else t_sample,
            "cosine": float(sim),  # Similarità coseno
            "covered": bool(sim >= threshold)  # Se il post è coperto
        })
    
    # Converte in DataFrame
    df_matches = pd.DataFrame(matches_data)
    
    # Salva nella cache su disco
    save_cache(user_id, source_type, threshold, model_name, metrics, df_matches)
    
    return metrics, df_matches, sim_matrix


@st.cache_data  # Cache Streamlit
def sensitivity_analysis_text(texts_full, text_sample, model_name="nomic-ai/modernbert-embed-base"):
    """
    Analisi di sensibilità per la copertura testuale.
    
    Varia la soglia da 0.0 a 1.0 e calcola F1, Precision e Recall
    per ogni valore, permettendo di visualizzare l'effetto della soglia.
    
    Args:
        texts_full (list): Lista dei post grezzi
        text_sample: Narrativa (stringa o lista)
        model_name (str): Nome del modello di embedding
        
    Returns:
        dict: Dizionario con 'thresholds', 'f1', 'precision', 'recall'
              o None se i dati sono insufficienti
    """
    # Verifica dati sufficienti
    if not texts_full or not text_sample:
        return None
    
    # Carica modello e genera embedding
    model = get_model(model_name)
    emb_full = embed_texts(model, texts_full)
    
    # Gestisce il formato del testo campione
    if isinstance(text_sample, str):
        texts_sample = [text_sample]
    else:
        texts_sample = text_sample
    
    emb_sample = embed_texts(model, texts_sample)
    # Calcola matrice di similarità
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # Range di soglie da testare
    threshold_range = np.arange(0.0, 1.05, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    # Per ogni soglia, calcola le metriche
    for t in threshold_range:
        matches_full_t = (sim_matrix.max(axis=1) >= t)  # Post coperti a questa soglia
        matches_sample_t = (sim_matrix.max(axis=0) >= t)  # Componenti corrispondenti

        tp_rec = matches_full_t.sum()
        tp_prec = matches_sample_t.sum()

        rec = tp_rec / len(texts_full) if len(texts_full) > 0 else 0
        prec = tp_prec / len(texts_sample) if len(texts_sample) > 0 else 0
        
        # Calcola F1
        if (prec + rec) > 0:
            score = 2 * prec * rec / (prec + rec)
        else:
            score = 0.0
        
        f1_scores.append(score)
        precisions.append(prec)
        recalls.append(rec)
    
    # Restituisce i dati per il grafico
    return {
        "thresholds": threshold_range,
        "f1": f1_scores,
        "precision": precisions,
        "recall": recalls
    }
