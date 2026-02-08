import numpy as np
import streamlit as st
import os
import json
from p0_global import data

# --- 3. Logica Segmentazione (Top-Down Piecewise Linear) ---
def point_line_distance(x0, y0, x1, y1, x2, y2):
    """Distanza dal punto (x0,y0) alla linea passante per (x1,y1) e (x2,y2)."""
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if den == 0:
        return 0.0
    return num / den

def get_segment_error(dates_num, scores, i_start, i_end):
    """Trova il punto con distanza massima dal segmento di linea."""
    if i_end <= i_start + 1:
        return 0.0, None
    
    x1, y1 = dates_num[i_start], scores[i_start]
    x2, y2 = dates_num[i_end], scores[i_end]
    
    max_dist = -1.0
    max_idx = None
    
    for i in range(i_start + 1, i_end):
        d = point_line_distance(dates_num[i], scores[i], x1, y1, x2, y2)
        if d > max_dist:
            max_dist = d
            max_idx = i
            
    if max_idx is None or max_dist < 0:
        return 0.0, None
    return max_dist, max_idx

@st.cache_data
def segment_time_series(series, k_segments=10):
    """
    Divide la serie temporale in K segmenti lineari usando approccio Top-Down. Usa cache su disco.
    """
    # Abbiamo bisogno di uno user_id qui. Poiché series non ce l'ha, potremmo aver bisogno di una chiave migliore.
    # Ma per ora, assumiamo che se chiamato da app.py, potremmo averlo o facciamo hash della serie
    import hashlib
    series_sig = hashlib.md5(str(series.values).encode()).hexdigest()[:10]
    
    cache_path = os.path.join(data.SEGMENT_CACHE_DIR, f"seg_{series_sig}_k{k_segments}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass

    if len(series) < 2:
        return []
    
    dates = series.index
    # Converti date in numerico (giorni dall'inizio) per calcolo geometria
    x = (dates - dates.min()).days.values.astype(float)
    y = series.values.astype(float)
    n = len(x)
    
    if n < k_segments:
        k_segments = n - 1
    
    segments = [(0, n-1)]
    
    while len(segments) < k_segments:
        best_err = 0.0
        best_split_idx = None
        best_seg_idx = None
        
        for idx, (stats, end) in enumerate(segments):
            err, split_idx = get_segment_error(x, y, stats, end)
            if err > best_err:
                best_err = err
                best_split_idx = split_idx
                best_seg_idx = idx
        
        if best_split_idx is None or best_err == 0:
            break
            
        org_start, org_end = segments[best_seg_idx]
        segments.pop(best_seg_idx)
        segments.insert(best_seg_idx, (best_split_idx, org_end))
        segments.insert(best_seg_idx, (org_start, best_split_idx))
        
    segments.sort(key=lambda s: s[0])
    
    results = []
    for start, end in segments:
        results.append({
            'start_date': str(dates[start].date()),
            'end_date': str(dates[end].date()),
            'start_val': float(y[start]),
            'end_val': float(y[end])
        })
    
    # Salva su disco
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    except:
        pass
        
    return results
