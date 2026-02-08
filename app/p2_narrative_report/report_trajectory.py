import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from p0_global import data

TRAJECTORY_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "reports", "trajectory")

def get_trajectory_path(user_id):
    return os.path.join(TRAJECTORY_REPORT_DIR, f"{user_id}.json")

def load_trajectory_report(user_id):
    path = get_trajectory_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None

def build_prompt_trajectory(posts, timestamps, mean_scores, phase, delta):
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
All posts belong exclusively to Phase $n$ of the user’s timeline.

Analyze the posts $p$ as a time series and describe the user’s emotional state across the entire period represented by Phase $n$.

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
    return prompt.strip()

def build_prompt_trajectory_summary(list_of_phase):
    prompt = f"""
Inputs:

Phase analyses ($P$): {list_of_phase}

Task:

You are given a chronologically ordered list of phase-level analytical narratives $P$, each describing the user’s emotional state during a specific phase of their timeline.

Analyze these phase summaries as a higher-level temporal sequence and produce an integrated summary of the user’s overall emotional trajectory across all phases.

Base the summary on the continuities, shifts, and inflection points described in the phase narratives, explaining how emotional tone, intensity, stability, and outlook evolve from earlier to later phases.

Identify recurring emotional patterns and highlight any phases that represent meaningful changes in direction, escalation, attenuation, or stabilization of emotional distress, as described in the inputs.

Synthesize the information into a single coherent narrative that reflects the longitudinal emotional trajectory, without introducing new interpretations, diagnoses, or assumptions beyond what is supported by the provided phase analyses.

Output:

Produce exactly one integrative analytical paragraph summarizing the user’s emotional trajectory across all phases.

Constraints:
The paragraph MUST be no longer than 200 words.
Do NOT restate individual phase labels verbatim.
Do NOT introduce timestamps or dates not already implied by the phase ordering.
Do NOT use bullet points, lists, or section headings.
Do NOT include meta-commentary or references to the task.
Do NOT introduce clinical or diagnostic conclusions.
        

"""
    return prompt.strip()

def ask_chatgpt(client, prompt):
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def ask_phase(posts, timestamps, mean_score, breaks, client):
    # Assicurati che timestamps e posts siano pd.Series
    timestamps = pd.to_datetime(pd.Series(timestamps))
    posts = pd.Series(posts)
    
    phases = []

    for i in range(len(breaks) - 1):
        # NOTA: Nel notebook accede a breaks['Date'][i], assumendo indicizzazione intera o lista.
        # breaks qui è un DataFrame. usa .iloc per cautela se index non è 0..N, ma df generato ha index default.
        start_date = pd.to_datetime(breaks['Date'].iloc[i])
        end_date = pd.to_datetime(breaks['Date'].iloc[i+1])
        
        mask = (timestamps >= start_date) & (timestamps <= end_date)
        posts_in_phase = posts[mask]
        timestamps_in_phase = timestamps[mask]
        
        # NOTA: Il notebook usa breaks['score_smooth'][i]
        delta = float(breaks['score_smooth'].iloc[i+1] - breaks['score_smooth'].iloc[i])
        
        # NOTA: Il notebook chiama build_prompt_trajectory con 5 argomenti
        # Nel notebook: build_prompt_trajectory(posts.tolist(), timestamps.tolist(), mean_score, i+1, delta)
        # Dobbiamo corrispondere a questo.
        # mean_score nel notebook è la serie completa.
        prompt_trajectory = build_prompt_trajectory(posts_in_phase.tolist(), 
                                                    timestamps_in_phase.tolist(),
                                                    mean_score, i+1, delta)
        phases.append(ask_chatgpt(client, prompt_trajectory))
    
    return phases

def ask_trajectory_summary(list_of_phase, client):
    if not list_of_phase:
        return "No phases analyzed."
    
    prompt_summary = build_prompt_trajectory_summary(list_of_phase)
    return ask_chatgpt(client, prompt_summary)


def generate_trajectory_report(user_id, user_df, segments, api_key, output_txt_path=None):
    """
    Genera Fase-per-Fase e Sommario Generale. Salva in cache JSON e opzionalmente TXT.
    """
    if not os.path.exists(TRAJECTORY_REPORT_DIR):
        os.makedirs(TRAJECTORY_REPORT_DIR, exist_ok=True)
        
    # 1. Controlla Cache
    cached = load_trajectory_report(user_id)
    if cached:
        return cached, True

    if not api_key:
        return None, False

    client = OpenAI(api_key=api_key)
    
    # --- Prepara Dati per corrispondere input Notebook ---
    
    # 1. post e timestamp
    posts = user_df["Text"]
    timestamps = user_df["Date"]
    
    # 2. mean_score (Medie giornaliere)
    # Usa data.calculate_daily_risk per mimare l'aggregazione del notebook
    # Notebook: mean_score = df.groupby("Date")["mean_score"].mean().sort_index()
    # verifica prima colonne user_df
    mean_score = pd.Series(dtype=float)
    if 'Prob_Severe_Depressed' in user_df.columns and 'Prob_Moderate_Depressed' in user_df.columns:
        # data.calculate_daily_risk restituisce Series indicizzata per Data, valori sono medie
        mean_score = data.calculate_daily_risk(user_df)

    # 3. breaks (DataFrame con 'Date' e 'score_smooth')
    # Usa lista segmenti per ricostruire la struttura 'breaks'
    breaks_data = []
    if segments:
        # Primo punto (inizio del primo segmento)
        breaks_data.append({
            'Date': segments[0]['start_date'],
            'score_smooth': segments[0]['start_val']
        })
        # Punti successivi (fine di ogni segmento)
        for seg in segments:
            breaks_data.append({
                'Date': seg['end_date'],
                'score_smooth': seg['end_val']
            })
            
    breaks = pd.DataFrame(breaks_data)
    
    # --- Esecuzione ---
    
    results = {}
    phase_narratives = []
    phase_details = []
    
    try:
        if not breaks.empty and len(breaks) > 1:
            # Chiama ask_phase esattamente come nel notebook (adattato per arg client)
            phase_narratives = ask_phase(posts, timestamps, mean_score, breaks, client)
            
            # Ricostruisci phase_details per UI App dalle narrative e segmenti
            # Assumiamo mappatura 1-a-1 tra segmenti e fasi generate
            for idx, narrative in enumerate(phase_narratives):
                if idx < len(segments):
                    seg = segments[idx]
                    phase_details.append({
                        "phase_num": idx+1,
                        "start_date": str(seg['start_date']),
                        "end_date": str(seg['end_date']),
                        "narrative": narrative,
                        "delta": seg['end_val'] - seg['start_val']
                    })
        else:
             phase_narratives = []
             phase_details = []

        # Chiama ask_summary
        trajectory_summary = ask_trajectory_summary(phase_narratives, client)
        
        results = {
            'phases': phase_details,
            'trajectory_summary': trajectory_summary
        }
        
    except Exception as e:
        results = {
            'phases': [],
            'trajectory_summary': f"Error generating report: {e}"
        }

    # Salva in Cache JSON
    with open(get_trajectory_path(user_id), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Salvataggio TXT opzionale (logica dal notebook)
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for item in phase_narratives:
                f.write(item + "\n\n")
            f.write("\nOverall trajectory:\n" + results['trajectory_summary'])
            
    return results, False
