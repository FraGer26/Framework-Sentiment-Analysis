# =============================================================================
# MODULO: topic_coverage.py
# DESCRIZIONE: Calcola la copertura dei topic (Topic Coverage) confrontando
#              i topic estratti dai post grezzi (ground truth) con quelli
#              estratti dalle narrative. Usa embedding e similarità coseno
#              per determinare quanto i topic candidati coprono quelli di riferimento.
# =============================================================================

import numpy as np  # Libreria per calcoli numerici e array
import pandas as pd  # Libreria per manipolazione dati tabulari
from sklearn.metrics.pairwise import cosine_similarity  # Funzione per calcolare la similarità coseno
import streamlit as st  # Framework per interfaccia web
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from p6_text_coverage import embedding_utils  # Modulo per utilità di embedding
from p4_topic_analysis import topic_model  # Modulo per l'estrazione dei topic
from p6_text_coverage.embedding_utils import get_model, embed_texts  # Funzioni per embedding

# Directory dove vengono salvati i risultati della copertura in cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "coverage")


def get_cache_path(user_id, source_type, threshold, model_name):
    """
    Costruisce il percorso del file cache per una specifica configurazione.
    
    Il nome del file include tutti i parametri per garantire univocità:
    user_id, tipo sorgente, soglia di similarità e nome del modello.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente candidata (es. "narrative_base")
        threshold (float): Soglia di similarità usata
        model_name (str): Nome del modello di embedding
        
    Returns:
        str: Percorso assoluto del file cache
    """
    # Sostituisce le barre nel nome del modello per compatibilità con i percorsi file
    filename = f"topic_cov_{user_id}_{source_type}_{threshold}_{model_name.replace('/', '_')}.json"
    return os.path.join(CACHE_DIR, filename)


def check_cache(user_id, source_type, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    """
    Verifica se i risultati della copertura sono in cache.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente candidata
        threshold (float): Soglia di similarità (default: 0.75)
        model_name (str): Nome del modello di embedding
        
    Returns:
        bool: True se il file cache esiste
    """
    path = get_cache_path(user_id, source_type, threshold, model_name)
    return os.path.exists(path)


def load_cache(user_id, source_type, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    """
    Carica i risultati della copertura dalla cache.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente candidata
        threshold (float): Soglia di similarità
        model_name (str): Nome del modello di embedding
        
    Returns:
        tuple: (metriche, df_corrispondenze, None)
               - metriche: dict con precision, recall, f1
               - df_corrispondenze: DataFrame con i dettagli delle corrispondenze
    """
    path = get_cache_path(user_id, source_type, threshold, model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Carica il JSON dalla cache
        
        # Ricostruisce il DataFrame dalla lista di dizionari salvata
        df_matches = pd.DataFrame(data["matches_data"])
        
        # Restituisce metriche, DataFrame e None (la matrice non viene salvata in cache)
        return data["metrics"], df_matches, None
    return None, None, None  # Restituisce None se la cache non esiste


def save_cache(user_id, source_type, threshold, model_name, metrics, df_matches):
    """
    Salva i risultati della copertura nella cache su disco.
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente candidata
        threshold (float): Soglia di similarità
        model_name (str): Nome del modello di embedding
        metrics (dict): Metriche calcolate (precision, recall, f1)
        df_matches (pd.DataFrame): DataFrame con i dettagli delle corrispondenze
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    path = get_cache_path(user_id, source_type, threshold, model_name)
    
    # Prepara i dati per la serializzazione JSON
    data = {
        "metrics": metrics,  # Dizionario delle metriche
        "matches_data": df_matches.to_dict(orient="records")  # DataFrame come lista di dizionari
    }
    
    # Salva il file JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


@st.cache_data  # Cache Streamlit per evitare ricalcoli
def calculate_coverage_metrics(user_id, source_type, topics_full, topics_sample, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    """
    Calcola le metriche di copertura (Precision, Recall, F1) dei topic.
    
    Confronta i topic di riferimento (full = ground truth dai post grezzi)
    con i topic candidati (sample = dalle narrative) usando embedding
    e similarità coseno.
    
    Logica:
    - Recall: quanti topic di riferimento sono "coperti" da almeno un candidato
    - Precision: quanti topic candidati corrispondono ad almeno un riferimento
    - F1: media armonica di Precision e Recall
    
    Args:
        user_id: ID dell'utente
        source_type (str): Tipo di sorgente candidata
        topics_full (list): Lista topic di riferimento (ground truth)
        topics_sample (list): Lista topic candidati (dalla narrativa)
        threshold (float): Soglia minima di similarità per considerare una corrispondenza
        model_name (str): Nome del modello di embedding
        
    Returns:
        tuple: (metriche, df_corrispondenze, matrice_similarità)
    """
    # Verifica che entrambe le liste contengano topic
    if not topics_full or not topics_sample:
        return None, None, None

    # Carica il modello di embedding (con cache Streamlit)
    model = get_model(model_name)
    
    # Genera gli embedding per tutti i topic
    emb_full = embed_texts(model, topics_full)  # Embedding dei topic di riferimento
    emb_sample = embed_texts(model, topics_sample)  # Embedding dei topic candidati
    
    # Calcola la matrice di similarità coseno (n_full x n_sample)
    # Ogni cella [i,j] contiene la similarità tra topic_full[i] e topic_sample[j]
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # ==========================================================================
    # CALCOLO CORRISPONDENZE
    # ==========================================================================
    # Per ogni topic di riferimento, trova il candidato più simile
    # Se la similarità massima >= threshold, il topic è "coperto"
    matches_full = (sim_matrix.max(axis=1) >= threshold)  # Bool per ogni topic full
    # Per ogni topic candidato, verifica se copre almeno un riferimento
    matches_sample = (sim_matrix.max(axis=0) >= threshold)  # Bool per ogni topic sample
    
    # Conta i veri positivi per recall e precision
    tp_recall = matches_full.sum()  # Topic riferimento coperti
    tp_precision = matches_sample.sum()  # Topic candidati che corrispondono
    
    # Calcola le metriche
    recall = tp_recall / len(topics_full) if topics_full else 0.0  # Copertura dei riferimenti
    precision = tp_precision / len(topics_sample) if topics_sample else 0.0  # Precisione dei candidati
    # F1: media armonica di precision e recall (evita divisione per zero)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Assembla il dizionario delle metriche
    metrics = {
        "precision": precision,  # Quanto sono rilevanti i candidati
        "recall": recall,  # Quanto i riferimenti sono coperti
        "f1": f1,  # Score bilanciato
        "threshold": threshold,  # Soglia usata
        "tp_recall": int(tp_recall),  # Veri positivi per recall
        "tp_precision": int(tp_precision),  # Veri positivi per precision
        "n_full": len(topics_full),  # Numero topic riferimento
        "n_sample": len(topics_sample)  # Numero topic candidati
    }
    
    # ==========================================================================
    # DETTAGLI CORRISPONDENZE
    # Per ogni topic di riferimento, trova il miglior candidato
    # ==========================================================================
    matches_data = []
    for i, t_full in enumerate(topics_full):
        j_best = sim_matrix[i].argmax()  # Indice del candidato più simile
        sim = sim_matrix[i, j_best]  # Similarità massima
        t_sample = topics_sample[j_best]  # Testo del candidato più simile
        
        matches_data.append({
            "full_topic": t_full,  # Topic di riferimento
            "sample_topic": t_sample,  # Miglior candidato
            "cosine": float(sim),  # Similarità coseno
            "matched": bool(sim >= threshold)  # Se supera la soglia
        })
    
    # Converte in DataFrame
    df_matches = pd.DataFrame(matches_data)
    
    # Salva i risultati nella cache su disco
    save_cache(user_id, source_type, threshold, model_name, metrics, df_matches)
    
    return metrics, df_matches, sim_matrix


@st.cache_data  # Cache Streamlit per evitare ricalcoli
def sensitivity_analysis(topics_full, topics_sample, model_name="nomic-ai/modernbert-embed-base"):
    """
    Analisi di sensibilità: calcola F1, Precision e Recall per diverse soglie.
    
    Varia la soglia da 0.0 a 1.0 con step 0.05 e calcola le metriche
    per ogni valore, permettendo di visualizzare come le prestazioni
    cambiano al variare della soglia di similarità.
    
    Args:
        topics_full (list): Lista topic di riferimento
        topics_sample (list): Lista topic candidati
        model_name (str): Nome del modello di embedding
        
    Returns:
        dict: Dizionario con chiavi 'thresholds', 'f1', 'precision', 'recall'
              o None se i dati sono insufficienti
    """
    # Verifica che ci siano dati sufficienti
    if not topics_full or not topics_sample:
        return None
    
    # Carica il modello e genera gli embedding
    model = get_model(model_name)
    emb_full = embed_texts(model, topics_full)
    emb_sample = embed_texts(model, topics_sample)
    
    # Calcola la matrice di similarità coseno
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # Range di soglie da testare: da 0.0 a 1.0 con step 0.05
    threshold_range = np.arange(0.0, 1.05, 0.05)
    f1_scores = []  # Lista per i valori F1
    precisions = []  # Lista per i valori di Precision
    recalls = []  # Lista per i valori di Recall
    
    # Per ogni soglia, calcola le metriche
    for t in threshold_range:
        # Trova le corrispondenze per questa soglia
        matches_full_t = (sim_matrix.max(axis=1) >= t)  # Topic riferimento coperti
        matches_sample_t = (sim_matrix.max(axis=0) >= t)  # Topic candidati corrispondenti

        tp_rec = matches_full_t.sum()  # Veri positivi recall
        tp_prec = matches_sample_t.sum()  # Veri positivi precision

        # Calcola recall e precision per questa soglia
        rec = tp_rec / len(topics_full) if len(topics_full) > 0 else 0
        prec = tp_prec / len(topics_sample) if len(topics_sample) > 0 else 0
        
        # Calcola F1 (media armonica)
        if (prec + rec) > 0:
            score = 2 * prec * rec / (prec + rec)
        else:
            score = 0.0
        
        # Accumula i risultati
        f1_scores.append(score)
        precisions.append(prec)
        recalls.append(rec)
    
    # Restituisce i dati per il grafico di sensibilità
    return {
        "thresholds": threshold_range,  # Array delle soglie
        "f1": f1_scores,  # Array dei valori F1
        "precision": precisions,  # Array dei valori Precision
        "recall": recalls  # Array dei valori Recall
    }
