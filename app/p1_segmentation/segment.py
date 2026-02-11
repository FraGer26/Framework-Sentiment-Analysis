# =============================================================================
# MODULO: segment.py
# DESCRIZIONE: Implementa la segmentazione della serie temporale del rischio
#              usando l'algoritmo Top-Down Piecewise Linear Approximation.
#              Divide la curva di rischio in K segmenti lineari che approssimano
#              le fasi comportamentali dell'utente.
# =============================================================================

import numpy as np  # Libreria per calcoli numerici e array
import streamlit as st  # Framework per interfaccia web (usato per cache)
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
import hashlib  # Libreria per generare hash MD5 (usata per cache)
from p0_global import data  # Modulo locale per costanti e percorsi cache


def point_line_distance(x0, y0, x1, y1, x2, y2):
    """
    Calcola la distanza perpendicolare di un punto da una retta.
    
    Usa la formula geometrica della distanza punto-retta nel piano:
    d = |( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 )| / sqrt((y2-y1)² + (x2-x1)²)
    
    Questa funzione è il cuore dell'algoritmo di segmentazione: determina
    quanto un punto si discosta dalla linea retta tra inizio e fine segmento.
    
    Args:
        x0, y0: Coordinate del punto da misurare
        x1, y1: Coordinate del primo estremo della retta
        x2, y2: Coordinate del secondo estremo della retta
        
    Returns:
        float: Distanza perpendicolare del punto dalla retta
    """
    # Calcola il numeratore della formula: valore assoluto del prodotto vettoriale
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    # Calcola il denominatore: lunghezza del segmento
    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    # Gestisce il caso degenere (segmento di lunghezza zero)
    if den == 0:
        return 0.0
    # Restituisce la distanza perpendicolare
    return num / den


def get_segment_error(dates_num, scores, i_start, i_end):
    """
    Trova il punto con la massima distanza dal segmento lineare.
    
    Scorre tutti i punti tra i_start e i_end e trova quello che
    si discosta maggiormente dalla retta che unisce gli estremi.
    Questo punto è il candidato ideale per dividere il segmento.
    
    Args:
        dates_num: Array di date convertite in numeri (giorni)
        scores: Array dei valori di rischio
        i_start: Indice del punto iniziale del segmento
        i_end: Indice del punto finale del segmento
        
    Returns:
        tuple: (massima_distanza, indice_punto_più_distante)
               Se il segmento è troppo corto, restituisce (0.0, None)
    """
    # Se il segmento ha meno di 3 punti, non c'è niente da dividere
    if i_end <= i_start + 1:
        return 0.0, None
    
    # Coordinate degli estremi del segmento
    x1, y1 = dates_num[i_start], scores[i_start]  # Punto iniziale
    x2, y2 = dates_num[i_end], scores[i_end]  # Punto finale
    
    # Inizializza la ricerca del massimo
    max_dist = -1.0  # Distanza massima trovata finora
    max_idx = None  # Indice del punto con distanza massima
    
    # Itera su tutti i punti intermedi (esclude gli estremi)
    for i in range(i_start + 1, i_end):
        # Calcola la distanza di questo punto dalla retta
        d = point_line_distance(dates_num[i], scores[i], x1, y1, x2, y2)
        # Aggiorna il massimo se questo punto è più distante
        if d > max_dist:
            max_dist = d
            max_idx = i
    
    # Gestisce caso in cui non è stato trovato nessun punto valido
    if max_idx is None or max_dist < 0:
        return 0.0, None
    
    # Restituisce la distanza massima e l'indice del punto corrispondente
    return max_dist, max_idx


@st.cache_data  # Decoratore Streamlit: salva il risultato in cache
def segment_time_series(series, k_segments=10):
    """
    Divide la serie temporale in K segmenti lineari usando l'algoritmo Top-Down.
    
    L'algoritmo funziona così:
    1. Inizia con un singolo segmento che copre tutta la serie
    2. Per ogni segmento, trova il punto che si discosta di più dalla linea retta
    3. Divide il segmento nel punto di massimo errore
    4. Ripete fino a ottenere K segmenti
    
    Questo approccio identifica automaticamente i "breakpoint" dove il trend
    del rischio cambia direzione, rivelando le fasi comportamentali.
    
    I risultati vengono salvati in cache su disco usando un hash MD5
    dei valori della serie come chiave univoca.
    
    Args:
        series (pd.Series): Serie temporale del rischio (indice=date, valori=rischio)
        k_segments (int): Numero di segmenti desiderati (default: 10)
        
    Returns:
        list: Lista di dizionari, ogni segmento ha:
              - start_date: Data inizio segmento
              - end_date: Data fine segmento
              - start_val: Valore rischio a inizio segmento
              - end_val: Valore rischio a fine segmento
    """
    # Genera un hash MD5 dei valori come identificatore univoco per la cache
    # Prende solo i primi 10 caratteri per brevità
    series_sig = hashlib.md5(str(series.values).encode()).hexdigest()[:10]
    
    # Costruisce il percorso del file cache per questa specifica serie
    cache_path = os.path.join(data.SEGMENT_CACHE_DIR, f"seg_{series_sig}_k{k_segments}.json")
    
    # Prova a caricare i segmenti dalla cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)  # Restituisce i segmenti dalla cache
        except:
            pass  # Se il caricamento fallisce, ricalcola

    # Se la serie ha meno di 2 punti, non è possibile segmentare
    if len(series) < 2:
        return []
    
    # Estrae le date dalla serie
    dates = series.index
    
    # Converte le date in valori numerici (giorni dall'inizio)
    # Necessario per il calcolo geometrico della distanza punto-retta
    x = (dates - dates.min()).days.values.astype(float)
    # Converte i valori di rischio in array float
    y = series.values.astype(float)
    # Numero totale di punti nella serie
    n = len(x)
    
    # Riduce il numero di segmenti se la serie è troppo corta
    if n < k_segments:
        k_segments = n - 1
    
    # ==========================================================================
    # ALGORITMO TOP-DOWN
    # Inizia con un segmento unico e lo divide iterativamente
    # ==========================================================================
    
    # Inizializza con un singolo segmento che copre tutta la serie
    # Ogni segmento è una tupla (indice_inizio, indice_fine)
    segments = [(0, n-1)]
    
    # Continua a dividere fino a raggiungere K segmenti
    while len(segments) < k_segments:
        best_err = 0.0  # Errore massimo trovato tra tutti i segmenti
        best_split_idx = None  # Indice del punto dove dividere
        best_seg_idx = None  # Indice del segmento da dividere
        
        # Per ogni segmento esistente, trova il punto di massimo errore
        for idx, (stats, end) in enumerate(segments):
            err, split_idx = get_segment_error(x, y, stats, end)
            # Aggiorna il migliore se questo segmento ha un errore maggiore
            if err > best_err:
                best_err = err
                best_split_idx = split_idx
                best_seg_idx = idx
        
        # Se non ci sono più punti da dividere, interrompe l'algoritmo
        if best_split_idx is None or best_err == 0:
            break
        
        # Divide il segmento con errore massimo in due sotto-segmenti
        org_start, org_end = segments[best_seg_idx]  # Estremi del segmento originale
        segments.pop(best_seg_idx)  # Rimuove il segmento originale
        # Inserisce i due nuovi sotto-segmenti al posto di quello rimosso
        segments.insert(best_seg_idx, (best_split_idx, org_end))  # Seconda metà
        segments.insert(best_seg_idx, (org_start, best_split_idx))  # Prima metà
    
    # Ordina i segmenti per posizione temporale (dal più vecchio al più recente)
    segments.sort(key=lambda s: s[0])
    
    # ==========================================================================
    # COSTRUZIONE RISULTATO
    # Converte gli indici numerici in date e valori leggibili
    # ==========================================================================
    results = []
    for start, end in segments:
        results.append({
            'start_date': str(dates[start].date()),  # Data inizio come stringa
            'end_date': str(dates[end].date()),  # Data fine come stringa
            'start_val': float(y[start]),  # Valore rischio a inizio segmento
            'end_val': float(y[end])  # Valore rischio a fine segmento
        })
    
    # Salva i risultati nella cache su disco
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)  # Salva come JSON con indentazione
    except:
        pass  # Ignora errori di salvataggio
    
    # Restituisce la lista dei segmenti
    return results
