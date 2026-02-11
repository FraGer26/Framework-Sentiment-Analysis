# =============================================================================
# MODULO: gpt_evaluator.py
# DESCRIZIONE: Implementa il sistema di valutazione cieca (blind A/B test)
#              dei report narrativi usando un LLM come giudice imparziale.
#              Confronta il report Base con il report Trajectory su 5 criteri
#              e genera statistiche aggregate e riepiloghi qualitativi.
# =============================================================================

import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
import random  # Libreria per randomizzazione (test A/B cieco)
import streamlit as st  # Framework per interfaccia web
from openai import OpenAI  # Client ufficiale per le API OpenAI
import hashlib  # Libreria per generare hash (disponibile)
import pandas as pd  # Libreria per manipolazione dati tabulari

# Directory dove vengono salvate le valutazioni in cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "evaluations")


def get_aggregate_stats():
    """
    Aggrega le statistiche da tutte le valutazioni salvate in cache.
    
    Scorre tutti i file di valutazione nella directory cache,
    estrae i punteggi per ogni criterio e calcola le medie,
    e conta le preferenze complessive (Base vs Trajectory vs Tie).
    
    Returns:
        dict: Dizionario con:
              - 'df': DataFrame con i punteggi medi per criterio
              - 'total_evals': Numero totale di valutazioni
              - 'preferences': Conteggio preferenze {Base, Trajectory, Tie}
              Restituisce None se non ci sono valutazioni
    """
    # Verifica che la directory cache esista
    if not os.path.exists(CACHE_DIR):
        return None
    
    # Elenca tutti i file di valutazione (formato: eval_*.json)
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("eval_") and f.endswith(".json")]
    
    # Se non ci sono file, restituisce None
    if not files:
        return None
    
    # ==========================================================================
    # ACCUMULO STATISTICHE
    # ==========================================================================
    stats = {}  # Dizionario per accumulare punteggi per ogni criterio
    total_evals = 0  # Contatore valutazioni totali
    preference_counts = {"Base": 0, "Trajectory": 0, "Tie": 0}  # Conteggio preferenze
    
    # Itera su ogni file di valutazione
    for filename in files:
        path = os.path.join(CACHE_DIR, filename)
        try:
            # Legge il file JSON della valutazione
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Estrae il risultato e la mappatura A/B
            result = data.get("result", {})  # Punteggi e preferenza
            mapping = data.get("mapping", {})  # Quale report è A e quale B
            
            # Salta file malformati
            if not result or not mapping:
                continue
            
            total_evals += 1  # Incrementa il contatore
            
            # =================================================================
            # IDENTIFICAZIONE REPORT
            # Determina quale ID (A o B) corrisponde al report base e trajectory
            # La mappatura indica: {"A": "base", "B": "trajectory"} o viceversa
            # =================================================================
            id_base = "A" if mapping.get("A") == "base" else "B"  # ID del report base
            id_traj = "A" if mapping.get("A") == "trajectory" else "B"  # ID del report trajectory
            
            # Estrae i punteggi per ogni report
            scores_base = result.get(f"Report_{id_base}", {})  # Punteggi report base
            scores_traj = result.get(f"Report_{id_traj}", {})  # Punteggi report trajectory
            
            # =================================================================
            # CONTEGGIO PREFERENZE
            # Determina quale report è stato preferito dal giudice LLM
            # =================================================================
            pref = result.get("Preferred_Report", "Tie")  # Preferenza espressa
            if pref == id_base:
                preference_counts["Base"] += 1  # Il giudice ha preferito il Base
            elif pref == id_traj:
                preference_counts["Trajectory"] += 1  # Il giudice ha preferito il Trajectory
            else:
                preference_counts["Tie"] += 1  # Pareggio
            
            # =================================================================
            # ACCUMULO PUNTEGGI PER CRITERIO
            # Somma i punteggi per calcolare le medie successive
            # =================================================================
            for criteria in scores_base.keys():
                # Inizializza la struttura per un nuovo criterio
                if criteria not in stats:
                    stats[criteria] = {"Base_Sum": 0, "Traj_Sum": 0, "Count": 0}
                
                # Somma i punteggi al totale corrente
                stats[criteria]["Base_Sum"] += scores_base.get(criteria, 0)
                stats[criteria]["Traj_Sum"] += scores_traj.get(criteria, 0)
                stats[criteria]["Count"] += 1  # Incrementa il contatore per la media
                
        except Exception as e:
            continue  # Salta file che non possono essere letti
    
    # Se non ci sono valutazioni valide, restituisce None
    if total_evals == 0:
        return None
    
    # ==========================================================================
    # COSTRUZIONE DATAFRAME RISULTATO
    # Calcola le medie e formatta per la visualizzazione
    # ==========================================================================
    rows = []
    for crit, val in stats.items():
        count = val["Count"]
        if count > 0:
            # Calcola le medie dividendo la somma per il conteggio
            avg_base = val["Base_Sum"] / count
            avg_traj = val["Traj_Sum"] / count
            
            rows.append({
                "Criterion": crit.replace("_", " "),  # Sostituisce underscore con spazi
                "Avg Score (Base)": f"{avg_base:.2f}",  # Media base con 2 decimali
                "Avg Score (Trajectory)": f"{avg_traj:.2f}"  # Media trajectory
            })
    
    # Converte la lista in DataFrame Pandas
    df = pd.DataFrame(rows)
    
    # Restituisce il dizionario con tutte le statistiche aggregate
    return {
        "df": df,  # DataFrame con punteggi medi
        "total_evals": total_evals,  # Numero totale valutazioni
        "preferences": preference_counts  # Conteggio preferenze
    }


# Nome del file cache per il riepilogo qualitativo
SUMMARY_CACHE_FILE = "eval_summary_cache.json"


def get_summary_cache_path():
    """
    Restituisce il percorso del file cache per il riepilogo qualitativo.
    
    Returns:
        str: Percorso assoluto del file JSON del riepilogo
    """
    return os.path.join(CACHE_DIR, SUMMARY_CACHE_FILE)


def load_qualitative_summary():
    """
    Carica il riepilogo qualitativo dalla cache.
    
    Returns:
        dict: Dizionario con le giustificazioni per criterio, o None se non esiste
    """
    path = get_summary_cache_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)  # Decodifica e restituisce il JSON
        except:
            pass  # File corrotto o illeggibile
    return None


def generate_qualitative_summary(api_key):
    """
    Genera un riepilogo qualitativo aggregato di tutte le valutazioni.
    
    Carica tutti i file di valutazione dalla cache, li invia a GPT
    chiedendo un confronto comparativo per ogni criterio, e salva
    il risultato in cache.
    
    Args:
        api_key (str): Chiave API OpenAI
        
    Returns:
        dict: Dizionario JSON con le giustificazioni per criterio, o None se fallisce
    """
    # Verifica che la directory cache esista
    if not os.path.exists(CACHE_DIR):
        return None
    
    # Carica tutti i file di valutazione (esclude il file riepilogo)
    files = [f for f in os.listdir(CACHE_DIR) if f.startswith("eval_") and f.endswith(".json")]
    all_judge_data = []  # Lista per accumulare i dati di tutte le valutazioni
    
    for filename in files:
        if filename == SUMMARY_CACHE_FILE: continue  # Salta il file riepilogo stesso
        path = os.path.join(CACHE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_judge_data.append(data)  # Aggiunge l'intera valutazione alla lista
        except:
            pass  # Salta file illeggibili
    
    # Se non ci sono dati, restituisce None
    if not all_judge_data:
        return None
    
    # Converte tutti i dati in una stringa JSON leggibile
    json_str = json.dumps(all_judge_data, indent=2)
    
    # Protezione contro input troppo lunghi per il contesto GPT
    # Tronca a 100.000 caratteri se necessario
    if len(json_str) > 100000:
        json_str = json_str[:100000] + "... (truncated)"
    
    # ==========================================================================
    # COSTRUZIONE PROMPT PER RIEPILOGO QUALITATIVO
    # Chiede a GPT di sintetizzare le valutazioni in giustificazioni comparative
    # ==========================================================================
    prompt = f"""
        Input:

        You are given a list of JSON objects representing evaluations: {json_str}

        Task:

        Instead of evaluating each report individually, aggregate all reports into a single comparison.
        Generate exactly one sentence of justification for each evaluation criterion that summarizes the relative strengths and weaknesses across all reports.
        Each justification must:
        - Compare the reports collectively
        - Reflect the relative scores qualitatively
        - Be concise, neutral, and technical
        Do not mention numeric values.
        Do not restate the criterion name.
        Do not add explanations beyond what is requested.

        Output:

        Return a single JSON object with the following structure:

        {{
            "Criterion_Justifications": {{
            "Trajectory_Coverage": "Aggregated comparative sentence across all reports",
            "Temporal_Coherence": "Aggregated comparative sentence across all reports",
            "Change_Point_Sensitivity": "Aggregated comparative sentence across all reports",
            "Segment_Level_Specificity": "Aggregated comparative sentence across all reports",
            "Overall_Preference": "Aggregated comparative sentence summarizing overall preference"
            }}
        }}
    """
    
    # Crea il client OpenAI
    client = OpenAI(api_key=api_key)
    
    try:
        # Chiama l'API GPT con il prompt di riepilogo
        response = client.chat.completions.create(
            model="gpt-5.1",  # Modello da utilizzare
            messages=[
                {"role": "system", "content": "You are an expert analyst of evaluation reports."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }  # Forza output JSON valido
        )
        
        # Estrae e decodifica il JSON dalla risposta
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # Salva il riepilogo in cache per usi futuri
        cache_path = get_summary_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)
        
        return result_json  # Restituisce il riepilogo generato
        
    except Exception as e:
        st.error(f"Error generating summary: {e}")  # Mostra errore nell'interfaccia
        return None


def get_cache_path(user_id, report_base, report_traj):
    """
    Costruisce il percorso del file cache per la valutazione di un utente.
    
    Il file è nominato con lo user_id per identificarlo univocamente.
    
    Args:
        user_id: ID dell'utente
        report_base: Testo del report base (non usato nel percorso)
        report_traj: Testo del report trajectory (non usato nel percorso)
        
    Returns:
        str: Percorso assoluto del file JSON della valutazione
    """
    filename = f"eval_{user_id}.json"
    return os.path.join(CACHE_DIR, filename)


def evaluate_reports(user_id, report_base, report_traj, api_key):
    """
    Esegue la valutazione cieca (blind A/B test) di due report narrativi.
    
    Il flusso del test cieco:
    1. Randomizza l'assegnazione A/B (il giudice non sa quale è Base e quale Trajectory)
    2. Invia entrambi i report a GPT con un prompt strutturato
    3. GPT valuta ogni report su 5 criteri con scala Likert 1-5
    4. GPT esprime una preferenza complessiva
    5. I risultati vengono salvati in cache con la mappatura A/B
    
    I 5 criteri di valutazione sono:
    1. Trajectory Coverage: copertura delle fasi principali
    2. Temporal Coherence: coerenza temporale della narrativa
    3. Change Point Sensitivity: identificazione dei punti di svolta
    4. Segment-Level Specificity: dettaglio specifico per fase
    5. Overall Preference: giudizio complessivo
    
    Args:
        user_id: ID dell'utente
        report_base (str): Testo del report base
        report_traj (str): Testo del report trajectory
        api_key (str): Chiave API OpenAI
        
    Returns:
        tuple: (risultato_json, mappatura)
               - risultato_json: dict con punteggi e preferenza
               - mappatura: dict che indica quale report era A e quale B
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Costruisce il percorso cache per questa valutazione
    cache_path = get_cache_path(user_id, report_base, report_traj)
    
    # ==========================================================================
    # CONTROLLA CACHE
    # Se la valutazione esiste già, la restituisce senza chiamare l'API
    # ==========================================================================
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # Restituisce il risultato e la mappatura dalla cache
            return cached_data["result"], cached_data["mapping"]
        except:
            pass  # Se la lettura fallisce, prosegue con il ricalcolo
    
    # ==========================================================================
    # RANDOMIZZAZIONE A/B (BLINDING)
    # Assegna casualmente i report alle etichette A e B
    # per garantire una valutazione imparziale
    # ==========================================================================
    if random.random() < 0.5:
        # Caso 1: A=Base, B=Trajectory
        report_A = report_base
        report_B = report_traj
        mapping = {
            "A": "base",
            "B": "trajectory"
        }
    else:
        # Caso 2: A=Trajectory, B=Base
        report_A = report_traj
        report_B = report_base
        mapping = {
            "A": "trajectory",
            "B": "base"
        }
    
    # ==========================================================================
    # COSTRUZIONE PROMPT DI VALUTAZIONE
    # Prompt strutturato che chiede al giudice LLM di valutare i due report
    # ==========================================================================
    user_prompt = f"""
Evaluate the following two reports according to the instructions below.

INSTRUCTIONS:
Carefully read Report A and Report B.
Evaluate each report independently using a 5-point Likert scale
(1 = Very poor, 5 = Excellent).

CRITERIA:
1. Trajectory Coverage
2. Temporal Coherence
3. Sensitivity to Change Points
4. Segment-Level Specificity
5. Overall Preference

CRITERIA DEFINITIONS:
1. Trajectory Coverage: how well the report captures the main phases of the user's mental-health–related history, rather than focusing only on isolated posts or a single time period.
2. Temporal Coherence: the extent to which the report clearly describes changes over time (e.g., worsening, improvement, stability) and maintains a logically consistent temporal narrative.
3. Sensitivity to Change Points:  how clearly the report identifies and explains important turning points in the trajectory, such as transitions from high to lower severity or vice versa.
4. Segment-Level Specificity: the amount of concrete, phase-specific detail included in the report, such as references to typical posts, themes, or coping strategies characteristic of each phase.
5. Overall Preference: an overall judgment of which report provides the most useful and coherent description of the user's depression-related trajectory.

OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "Report_A": {{
    "Trajectory_Coverage": X,
    "Temporal_Coherence": X,
    "Change_Point_Sensitivity": X,
    "Segment_Level_Specificity": X,
    "Overall_Preference": X
  }},
  "Report_B": {{
    "Trajectory_Coverage": X,
    "Temporal_Coherence": X,
    "Change_Point_Sensitivity": X,
    "Segment_Level_Specificity": X,
    "Overall_Preference": X
  }},
  "Preferred_Report": "A | B | Tie",
  "Criterion_Justifications": {{
    "Trajectory_Coverage": "Concise justification comparing A and B (2 sentences).",
    "Temporal_Coherence": "Concise justification comparing A and B (2 sentences).",
    "Change_Point_Sensitivity": "Concise justification comparing A and B (2 sentences).",
    "Segment_Level_Specificity": "Concise justification comparing A and B (2 sentences).",
    "Overall_Preference": "Concise justification comparing A and B (2 sentences)."
  }},

  "Rationale": "Concise justification (2–4 sentences)."
}}

REPORT A:
<<<
{report_A}
>>>

REPORT B:
<<<
{report_B}
>>>
"""

    # Crea il client OpenAI con la chiave API
    client = OpenAI(api_key=api_key)
    
    try:
        # Chiama l'API GPT con il prompt di valutazione
        response = client.chat.completions.create(
            model="gpt-5.1",  # Modello da utilizzare
            temperature=0.0,  # Temperatura 0 per massima riproducibilità
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert, impartial evaluator of mental-health trajectory reports. "
                        "Judge reports solely based on their content. "
                        "Do not infer their origin. Output only valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt  # Il prompt con i due report
                }
            ],
            response_format={ "type": "json_object" }  # Forza output JSON valido
        )
        
        # Estrae e decodifica il JSON dalla risposta
        content = response.choices[0].message.content
        result_json = json.loads(content)
        
        # Aggiunge la mappatura al risultato per trasparenza
        result_json["Report_Mapping"] = mapping
        
        # =================================================================
        # SALVATAGGIO IN CACHE
        # Salva sia il risultato che la mappatura per riferimento futuro
        # =================================================================
        cache_data = {
            "result": result_json,  # Punteggi e preferenza del giudice
            "mapping": mapping  # Quale report era A e quale B
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=4)
        
        # Restituisce il risultato e la mappatura
        return result_json, mapping
        
    except Exception as e:
        st.error(f"Error during evaluation: {e}")  # Mostra errore nell'interfaccia
        return None, None
