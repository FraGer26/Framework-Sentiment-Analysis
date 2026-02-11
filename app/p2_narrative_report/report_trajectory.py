# =============================================================================
# MODULO: report_trajectory.py
# DESCRIZIONE: Genera il report narrativo "Trajectory" per un utente tramite GPT.
#              A differenza del report Base, questo report utilizza la segmentazione
#              per analizzare ogni fase comportamentale separatamente e poi
#              produrre un riepilogo integrato dell'intera traiettoria emotiva.
# =============================================================================

import pandas as pd  # Libreria per manipolazione dati tabulari
import numpy as np  # Libreria per calcoli numerici
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from openai import OpenAI  # Client ufficiale per le API OpenAI
from p0_global import data  # Modulo locale per costanti e funzioni dati

# Directory dove vengono salvati i report trajectory in cache
TRAJECTORY_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "reports", "trajectory")


def get_trajectory_path(user_id):
    """
    Costruisce il percorso del file cache per il report trajectory di un utente.
    
    Args:
        user_id: ID dell'utente
        
    Returns:
        str: Percorso assoluto del file JSON del report
    """
    return os.path.join(TRAJECTORY_REPORT_DIR, f"{user_id}.json")


def load_trajectory_report(user_id):
    """
    Carica il report trajectory dalla cache su disco.
    
    Args:
        user_id: ID dell'utente
        
    Returns:
        dict: Dizionario con il report (fasi + riepilogo), o None se non esiste
    """
    # Costruisce il percorso e verifica esistenza
    path = get_trajectory_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)  # Decodifica il JSON
        except:
            return None  # File corrotto
    return None  # File non trovato


def build_prompt_trajectory(posts, timestamps, mean_scores, phase, delta):
    """
    Costruisce il prompt per l'analisi di una singola fase della traiettoria.
    
    Il prompt chiede a GPT di analizzare i post di una specifica fase,
    usando il punteggio medio giornaliero e il delta come segnali contestuali
    (non come criteri diagnostici).
    
    Args:
        posts: Lista dei testi dei post nella fase
        timestamps: Lista delle date dei post nella fase
        mean_scores: Serie dei punteggi medi giornalieri
        phase: Numero della fase (1-based)
        delta: Variazione del rischio tra inizio e fine fase
        
    Returns:
        str: Prompt formattato per GPT
    """
    # Template del prompt con variabili interpolate per questa specifica fase
    prompt = f"""
Inputs:

Posts ($p$): {posts}
Timestamps ($t$): {timestamps}
Daily mean scores ($c$): {mean_scores}
Phase number ($n$): {phase}
Delta value ($d$): {delta}

Task:

You are given a series of social media posts $p$ written by a single user, sorted by timestamp $t$.
Each post in $p$ is associated with a timestamp $t$. The series $c$ contains the daily mean scores for each day represented in $t$.
All posts belong exclusively to Phase $n$ of the user's timeline.

Analyze the posts $p$ as a time series and describe the user's emotional state across the entire period represented by Phase $n$.

Base the analysis on the explicit semantic, emotional, and expressive content of the posts.
Use the daily mean scores $c$ and the delta value $d$ strictly as auxiliary contextual signals to support interpretation of emotional intensity and directional tendency within Phase $n$. Do not treat them as labels, diagnoses, or primary decision criteria.

Identify any posts that exhibit depressive-leaning emotional expressions and explain why they are salient in comparison to other posts within the same phase.

Describe when these emotional expressions are most pronounced and explain how emotional tone, intensity, or outlook changes from the beginning to the end of Phase $n$, incorporating the general directional information conveyed by delta $d$ while allowing for gradual, uneven, or internally mixed emotional progression.

Synthesize the content of the posts into a single, coherent analytical narrative that reconstructs the emotional evolution contained entirely within Phase $n$, without extrapolation beyond the provided timestamps.

Output format (must be followed exactly):

Phase $n$ (from <computed_start_date> to <computed_end_date>):
 <one analytically coherent paragraph describing this phase>

Constraints:

Dates MUST be computed strictly from timestamps $t$.
Produce exactly one paragraph.
The paragraph MUST be no longer than 100 words.
Do NOT reference content, phases, or emotional states outside Phase $n$.
Do NOT introduce diagnostic or clinical conclusions.
Do NOT use bullet points, lists, or headings other than the exact label specified above.
Do NOT include meta-commentary, justifications, or references to the task.
"""
    # Rimuove spazi bianchi iniziali e finali
    return prompt.strip()


def build_prompt_trajectory_summary(list_of_phase):
    """
    Costruisce il prompt per il riepilogo integrato di tutte le fasi.
    
    Chiede a GPT di sintetizzare le narrative delle singole fasi in un
    unico paragrafo che descrive la traiettoria emotiva complessiva.
    
    Args:
        list_of_phase: Lista delle narrative delle singole fasi
        
    Returns:
        str: Prompt formattato per GPT
    """
    # Template del prompt per il riepilogo
    prompt = f"""
Inputs:

Phase analyses ($P$): {list_of_phase}

Task:

You are given a chronologically ordered list of phase-level analytical narratives $P$, each describing the user's emotional state during a specific phase of their timeline.

Analyze these phase summaries as a higher-level temporal sequence and produce an integrated summary of the user's overall emotional trajectory across all phases.

Base the summary on the continuities, shifts, and inflection points described in the phase narratives, explaining how emotional tone, intensity, stability, and outlook evolve from earlier to later phases.

Identify recurring emotional patterns and highlight any phases that represent meaningful changes in direction, escalation, attenuation, or stabilization of emotional distress, as described in the inputs.

Synthesize the information into a single coherent narrative that reflects the longitudinal emotional trajectory, without introducing new interpretations, diagnoses, or assumptions beyond what is supported by the provided phase analyses.

Output:

Produce exactly one integrative analytical paragraph summarizing the user's emotional trajectory across all phases.

Constraints:
The paragraph MUST be no longer than 200 words.
Do NOT restate individual phase labels verbatim.
Do NOT introduce timestamps or dates not already implied by the phase ordering.
Do NOT use bullet points, lists, or section headings.
Do NOT include meta-commentary or references to the task.
Do NOT introduce clinical or diagnostic conclusions.
        

"""
    # Rimuove spazi bianchi
    return prompt.strip()


def ask_chatgpt(client, prompt):
    """
    Invia un prompt a GPT e restituisce la risposta testuale.
    
    Args:
        client: Istanza del client OpenAI
        prompt (str): Prompt da inviare
        
    Returns:
        str: Testo della risposta generata dal modello
    """
    # Chiama l'API con il modello GPT specificato
    response = client.chat.completions.create(
        model="gpt-5.1",  # Modello da utilizzare
        messages=[{"role": "user", "content": prompt}]  # Prompt come messaggio utente
    )
    # Estrae il contenuto testuale dalla risposta
    return response.choices[0].message.content


def ask_phase(posts, timestamps, mean_score, breaks, client):
    """
    Genera la narrativa per ogni fase della traiettoria.
    
    Itera sui breakpoint della segmentazione, filtra i post per ogni fase,
    calcola il delta del rischio e chiama GPT per ogni fase.
    
    Args:
        posts: Serie dei testi dei post
        timestamps: Serie delle date dei post
        mean_score: Serie dei punteggi medi giornalieri
        breaks (pd.DataFrame): DataFrame con colonne 'Date' e 'score_smooth'
        client: Istanza del client OpenAI
        
    Returns:
        list: Lista di stringhe, ogni elemento è la narrativa di una fase
    """
    # Converte timestamps e posts in Serie Pandas per uniformità
    timestamps = pd.to_datetime(pd.Series(timestamps))
    posts = pd.Series(posts)
    
    phases = []  # Lista per accumulare le narrative delle fasi

    # Itera su coppie consecutive di breakpoint (definiscono le fasi)
    for i in range(len(breaks) - 1):
        # Estrae le date di inizio e fine fase dai breakpoint
        # Usa .iloc per accesso posizionale sicuro
        start_date = pd.to_datetime(breaks['Date'].iloc[i])  # Data inizio fase
        end_date = pd.to_datetime(breaks['Date'].iloc[i+1])  # Data fine fase
        
        # Crea la maschera per filtrare i post che cadono in questa fase
        mask = (timestamps >= start_date) & (timestamps <= end_date)
        posts_in_phase = posts[mask]  # Post filtrati per questa fase
        timestamps_in_phase = timestamps[mask]  # Date filtrate per questa fase
        
        # Calcola il delta: variazione del punteggio tra fine e inizio fase
        # Un delta positivo indica peggioramento, negativo miglioramento
        delta = float(breaks['score_smooth'].iloc[i+1] - breaks['score_smooth'].iloc[i])
        
        # Costruisce il prompt per questa fase specifica
        prompt_trajectory = build_prompt_trajectory(
            posts_in_phase.tolist(),  # Converte la Serie in lista Python
            timestamps_in_phase.tolist(),  # Converte le date in lista
            mean_score,  # Serie completa dei punteggi medi
            i+1,  # Numero della fase (1-based)
            delta  # Variazione del rischio
        )
        # Invia il prompt a GPT e aggiunge la risposta alla lista delle fasi
        phases.append(ask_chatgpt(client, prompt_trajectory))
    
    # Restituisce la lista delle narrative generate
    return phases


def ask_trajectory_summary(list_of_phase, client):
    """
    Genera il riepilogo integrato di tutte le fasi.
    
    Args:
        list_of_phase: Lista delle narrative delle singole fasi
        client: Istanza del client OpenAI
        
    Returns:
        str: Narrativa riassuntiva dell'intera traiettoria
    """
    # Se non ci sono fasi, restituisce un messaggio predefinito
    if not list_of_phase:
        return "No phases analyzed."
    
    # Costruisce il prompt per il riepilogo e lo invia a GPT
    prompt_summary = build_prompt_trajectory_summary(list_of_phase)
    return ask_chatgpt(client, prompt_summary)


def generate_trajectory_report(user_id, user_df, segments, api_key, output_txt_path=None):
    """
    Genera il report trajectory completo: analisi per fase + riepilogo.
    
    Flusso:
    1. Controlla cache
    2. Prepara i dati (post, date, punteggi, breakpoint)
    3. Genera narrativa per ogni fase
    4. Genera riepilogo integrato
    5. Salva in cache JSON e opzionalmente TXT
    
    Args:
        user_id: ID dell'utente
        user_df (pd.DataFrame): DataFrame con i dati dell'utente
        segments (list): Lista dei segmenti dalla segmentazione
        api_key (str): Chiave API OpenAI
        output_txt_path (str, optional): Percorso per export TXT
        
    Returns:
        tuple: (dizionario_risultato, bool_da_cache)
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(TRAJECTORY_REPORT_DIR):
        os.makedirs(TRAJECTORY_REPORT_DIR, exist_ok=True)
        
    # Fase 1: Controlla se il report è già in cache
    cached = load_trajectory_report(user_id)
    if cached:
        return cached, True  # Restituisce dalla cache

    # Se non c'è API key, non può generare
    if not api_key:
        return None, False

    # Crea il client OpenAI
    client = OpenAI(api_key=api_key)
    
    # ==========================================================================
    # PREPARAZIONE DATI
    # Replica la struttura delle variabili del notebook originale
    # ==========================================================================
    
    # Estrae post e timestamp dal DataFrame dell'utente
    posts = user_df["Text"]  # Testi dei post
    timestamps = user_df["Date"]  # Date dei post
    
    # Calcola i punteggi medi giornalieri usando la funzione di data.py
    # Formula: mean_score = media giornaliera di (2*Severe + Moderate)
    mean_score = pd.Series(dtype=float)  # Inizializza serie vuota
    if 'Prob_Severe_Depressed' in user_df.columns and 'Prob_Moderate_Depressed' in user_df.columns:
        mean_score = data.calculate_daily_risk(user_df)

    # Ricostruisce il DataFrame 'breaks' dalla lista dei segmenti
    # Ogni breakpoint ha una data e un valore di rischio smussato
    breaks_data = []
    if segments:
        # Aggiunge il punto iniziale del primo segmento
        breaks_data.append({
            'Date': segments[0]['start_date'],  # Data inizio
            'score_smooth': segments[0]['start_val']  # Valore rischio smussato
        })
        # Aggiunge il punto finale di ogni segmento (= inizio del successivo)
        for seg in segments:
            breaks_data.append({
                'Date': seg['end_date'],  # Data fine segmento
                'score_smooth': seg['end_val']  # Valore rischio smussato
            })
    
    # Converte la lista di dizionari in DataFrame Pandas
    breaks = pd.DataFrame(breaks_data)
    
    # ==========================================================================
    # GENERAZIONE REPORT
    # ==========================================================================
    
    results = {}  # Dizionario per i risultati finali
    phase_narratives = []  # Lista delle narrative dalle chiamate GPT
    phase_details = []  # Lista dettagliata con metadati per l'interfaccia
    
    try:
        # Verifica che ci siano almeno 2 breakpoint per definire una fase
        if not breaks.empty and len(breaks) > 1:
            # Genera la narrativa per ogni fase chiamando GPT
            phase_narratives = ask_phase(posts, timestamps, mean_score, breaks, client)
            
            # Costruisce i dettagli delle fasi per l'interfaccia utente
            # Associa ogni narrativa al segmento corrispondente
            for idx, narrative in enumerate(phase_narratives):
                if idx < len(segments):
                    seg = segments[idx]  # Segmento corrispondente
                    phase_details.append({
                        "phase_num": idx+1,  # Numero fase (1-based)
                        "start_date": str(seg['start_date']),  # Data inizio
                        "end_date": str(seg['end_date']),  # Data fine
                        "narrative": narrative,  # Narrativa generata da GPT
                        "delta": seg['end_val'] - seg['start_val']  # Variazione rischio
                    })
        else:
            # Se non ci sono abbastanza breakpoint, liste vuote
            phase_narratives = []
            phase_details = []

        # Genera il riepilogo integrato di tutte le fasi
        trajectory_summary = ask_trajectory_summary(phase_narratives, client)
        
        # Assembla il risultato finale
        results = {
            'phases': phase_details,  # Dettagli di ogni fase
            'trajectory_summary': trajectory_summary  # Riepilogo complessivo
        }
        
    except Exception as e:
        # In caso di errore, salva il messaggio
        results = {
            'phases': [],
            'trajectory_summary': f"Error generating report: {e}"
        }

    # Salva il report completo in cache JSON
    with open(get_trajectory_path(user_id), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Salvataggio opzionale come file TXT (replica il notebook)
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            # Scrive ogni narrativa di fase seguita da una riga vuota
            for item in phase_narratives:
                f.write(item + "\n\n")
            # Aggiunge il riepilogo complessivo alla fine
            f.write("\nOverall trajectory:\n" + results['trajectory_summary'])
    
    # Restituisce il risultato con flag False (non da cache)
    return results, False
