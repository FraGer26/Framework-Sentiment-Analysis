import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit as st
import os
import json
import hashlib
from embedding_utils import get_model, embed_texts

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "coverage")

def get_cache_path(user_id, source_type, threshold, model_name):
    # Predictable cache creation based on user and configuration
    # source_type should be "base" or "traj"
    filename = f"text_cov_{user_id}_{source_type}_{threshold}_{model_name.replace('/', '_')}.json"
    return os.path.join(CACHE_DIR, filename)

def check_cache(user_id, source_type, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    path = get_cache_path(user_id, source_type, threshold, model_name)
    return os.path.exists(path)

def load_cache(user_id, source_type, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    path = get_cache_path(user_id, source_type, threshold, model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df_matches = pd.DataFrame(data["matches_data"])
        return data["metrics"], df_matches, None
    return None, None, None

def save_cache(user_id, source_type, threshold, model_name, metrics, df_matches):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = get_cache_path(user_id, source_type, threshold, model_name)
    data = {
        "metrics": metrics,
        "matches_data": df_matches.to_dict(orient="records")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

@st.cache_data
def calculate_text_coverage_metrics(user_id, source_type, texts_full, text_sample, threshold=0.5, model_name="nomic-ai/modernbert-embed-base"):
    """
    Calculates coverage metrics of raw texts (full) against a summary text (sample).
    texts_full: List of strings (e.g., raw posts)
    text_sample: String (narrative) or List of strings
    """
    if not texts_full or not text_sample:
        return None, None, None

    model = get_model(model_name)
    
    # Embed Full Texts (Batch processing handled by sentence-transformers)
    emb_full = embed_texts(model, texts_full)
    
    # Handle Sample Text (Ensure it's a list)
    if isinstance(text_sample, str):
        texts_sample = [text_sample]
    elif isinstance(text_sample, list):
        texts_sample = text_sample
    else:
        return None, None, None
        
    emb_sample = embed_texts(model, texts_sample)
    
    # Cosine Similarity Matrix
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # Matches: For each full text, is there at least one sample text with sim >= threshold?
    # Axis 1 = max over columns (samples). If max >= threshold, the full text is "covered".
    matches_full = (sim_matrix.max(axis=1) >= threshold)
    
    # For each sample text, does it cover at least one full text? (Less critical for recall, but for precision)
    matches_sample = (sim_matrix.max(axis=0) >= threshold)
    
    tp_recall = matches_full.sum()
    tp_precision = matches_sample.sum()
    
    recall = tp_recall / len(texts_full) if texts_full else 0.0
    precision = tp_precision / len(texts_sample) if texts_sample else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp_recall": int(tp_recall),
        "tp_precision": int(tp_precision),
        "n_full": len(texts_full),
        "n_sample": len(texts_sample)
    }
    
    # Top Matches Details (e.g. top 50 to avoid huge tables)
    matches_data = []
    # We want to see which full text matches best with which sample text
    for i, t_full in enumerate(texts_full):
        j_best = sim_matrix[i].argmax()
        sim = sim_matrix[i, j_best]
        t_sample = texts_sample[j_best]
        
        # Only keep interesting ones or strictly top N? 
        # Let's keep all but sorting will be done in UI.
        # To save memory if texts_full is huge, we might limit this. 
        # But for n=1000 it's fine.
        
        matches_data.append({
            "full_text": t_full,
            "sample_component": t_sample[:100] + "..." if len(t_sample) > 100 else t_sample, # Truncate sample for display
            "cosine": float(sim),
            "covered": bool(sim >= threshold)
        })
        
    df_matches = pd.DataFrame(matches_data)
    
    save_cache(user_id, source_type, threshold, model_name, metrics, df_matches)
    
    return metrics, df_matches, sim_matrix

@st.cache_data
def sensitivity_analysis_text(texts_full, text_sample, model_name="nomic-ai/modernbert-embed-base"):
    """
    Returns data arrays for plotting F1 vs Threshold for Text Coverage.
    """
    if not texts_full or not text_sample:
        return None
        
    model = get_model(model_name)
    emb_full = embed_texts(model, texts_full)
    
    if isinstance(text_sample, str):
        texts_sample = [text_sample]
    else:
        texts_sample = text_sample
        
    emb_sample = embed_texts(model, texts_sample)
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    threshold_range = np.arange(0.0, 1.05, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    for t in threshold_range:
        matches_full_t = (sim_matrix.max(axis=1) >= t)
        matches_sample_t = (sim_matrix.max(axis=0) >= t) # Not strictly standard precision for text-to-text summary but useful proxy

        tp_rec = matches_full_t.sum()
        tp_prec = matches_sample_t.sum()

        rec = tp_rec / len(texts_full) if len(texts_full) > 0 else 0
        prec = tp_prec / len(texts_sample) if len(texts_sample) > 0 else 0
        
        if (prec + rec) > 0:
            score = 2 * prec * rec / (prec + rec)
        else:
            score = 0.0
            
        f1_scores.append(score)
        precisions.append(prec)
        recalls.append(rec)
        
    return {
        "thresholds": threshold_range,
        "f1": f1_scores,
        "precision": precisions,
        "recall": recalls
    }
