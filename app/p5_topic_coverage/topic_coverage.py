import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit as st
import os
import json
from p6_text_coverage import embedding_utils
from p4_topic_analysis import topic_model
from p6_text_coverage.embedding_utils import get_model, embed_texts

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "coverage")

def get_cache_path(user_id, source_type, threshold, model_name):
    # Predictable cache creation based on user and configuration
    filename = f"topic_cov_{user_id}_{source_type}_{threshold}_{model_name.replace('/', '_')}.json"
    return os.path.join(CACHE_DIR, filename)

def check_cache(user_id, source_type, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    path = get_cache_path(user_id, source_type, threshold, model_name)
    return os.path.exists(path)

def load_cache(user_id, source_type, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    path = get_cache_path(user_id, source_type, threshold, model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct DataFrame from list of dicts
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
def calculate_coverage_metrics(user_id, source_type, topics_full, topics_sample, threshold=0.75, model_name="nomic-ai/modernbert-embed-base"):
    """
    Calculates coverage metrics (Precision, Recall, F1) of sample topics against full topics.
    """
    if not topics_full or not topics_sample:
        return None, None, None # Adjusted to return 3 values similar to original signature expectation or internal usage

    model = get_model(model_name)
    
    emb_full = embed_texts(model, topics_full)
    emb_sample = embed_texts(model, topics_sample)
    
    # Cosine Similarity Matrix
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    # Matches
    matches_full = (sim_matrix.max(axis=1) >= threshold)
    matches_sample = (sim_matrix.max(axis=0) >= threshold)
    
    tp_recall = matches_full.sum()
    tp_precision = matches_sample.sum()
    
    recall = tp_recall / len(topics_full) if topics_full else 0.0
    precision = tp_precision / len(topics_sample) if topics_sample else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp_recall": int(tp_recall),
        "tp_precision": int(tp_precision),
        "n_full": len(topics_full),
        "n_sample": len(topics_sample)
    }
    
    # Match Details
    matches_data = []
    for i, t_full in enumerate(topics_full):
        j_best = sim_matrix[i].argmax()
        sim = sim_matrix[i, j_best]
        t_sample = topics_sample[j_best]
        
        matches_data.append({
            "full_topic": t_full,
            "sample_topic": t_sample,
            "cosine": float(sim),
            "matched": bool(sim >= threshold)
        })
        
    df_matches = pd.DataFrame(matches_data)
    
    # Save to disk cache
    save_cache(user_id, source_type, threshold, model_name, metrics, df_matches)
    
    return metrics, df_matches, sim_matrix

@st.cache_data
def sensitivity_analysis(topics_full, topics_sample, model_name="nomic-ai/modernbert-embed-base"):
    """
    Returns data arrays for plotting F1 vs Threshold.
    """
    if not topics_full or not topics_sample:
        return None
        
    model = get_model(model_name)
    emb_full = embed_texts(model, topics_full)
    emb_sample = embed_texts(model, topics_sample)
    sim_matrix = cosine_similarity(emb_full, emb_sample)
    
    threshold_range = np.arange(0.0, 1.05, 0.05)
    f1_scores = []
    precisions = []
    recalls = []
    
    for t in threshold_range:
        matches_full_t = (sim_matrix.max(axis=1) >= t)
        matches_sample_t = (sim_matrix.max(axis=0) >= t)

        tp_rec = matches_full_t.sum()
        tp_prec = matches_sample_t.sum()

        rec = tp_rec / len(topics_full) if len(topics_full) > 0 else 0
        prec = tp_prec / len(topics_sample) if len(topics_sample) > 0 else 0
        
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
