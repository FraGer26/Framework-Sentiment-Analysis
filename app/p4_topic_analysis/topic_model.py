# =============================================================================
# MODULO: topic_model.py
# DESCRIZIONE: Estrae argomenti (topic) da un testo usando GPT.
#              Ogni argomento è classificato come positivo, neutro o negativo
#              in base alla valenza emotiva dominante nel testo.
#              I risultati vengono salvati in cache per evitare chiamate API ripetute.
# =============================================================================

import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
import streamlit as st  # Framework per interfaccia web
from openai import OpenAI  # Client ufficiale per le API OpenAI

# Directory dove vengono salvati i topic estratti in cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "topics")


def build_prompt_granular(text):
    """
    Costruisce il prompt per l'estrazione granulare dei topic.
    
    Il prompt enfatizza la granularità: GPT deve estrarre argomenti
    specifici e concreti (6-15 parole ciascuno), evitando generalizzazioni.
    Ogni argomento viene classificato in positivo/neutro/negativo.
    
    Args:
        text (str): Testo da analizzare (post utente o narrativa)
        
    Returns:
        str: Prompt formattato per GPT
    """
    # Template del prompt con istruzioni dettagliate per l'estrazione
    prompt = f"""
Input
A text describing a user's experiences and emotional states over time.

TEXT:
{text}

Task
Extract the specific, concrete, and distinct topics that characterize the user's experience.
A topic is a detailed theme reflecting specific situations, symptoms, behaviors, or concerns described in the text.

CRITICAL INSTRUCTION: GRANULARITY OVER GENERALITY
- Do NOT generalize specific events into broad categories. Avoid generic labels.
- If the text mentions specific methods, explicit behaviors, or concrete triggers, include those precise details in the topic string.
- Preserving the specific context (the "who", "where", and "how") is required rather than using abstract nouns.
- Differentiate between similar themes if the underlying details or contexts differ.

Guidelines:
- Each topic must be a descriptive phrase (6-15 words) to ensure sufficient detail.
- Aim for 10-20 distinct topics to ensure full coverage of the narrative details.
- Use the exact terminology found in the text if it adds precision to the description.

Assign each topic to exactly one emotional valence category based on the dominant emotional tone across the entire text:
- positive: relief, improvement, support, coping, positive change;
- neutral: mixed, ambiguous, or emotionally balanced;
- negative: distress, difficulties, negative emotions.
If a topic includes both positive and negative aspects, classify it according to the prevailing valence.

Output
Return only a valid Python dictionary (JSON format) with exactly three keys:
{{
  "positivetopics": [...],
  "neutraltopics": [...],
  "negativetopics": [...]
}}

Each value must be a list of strings.
A topic may appear in only one list.
Do not include comments, explanations, or quotations outside the dictionary.
"""
    # Rimuove spazi bianchi iniziali e finali
    return prompt.strip()


def extract_topics(user_id, text, api_key, source_type="posts"):
    """
    Estrae i topic da un testo per un utente specifico.
    
    Flusso:
    1. Controlla se i topic sono in cache (per utente + sorgente)
    2. Se non in cache, chiama GPT per estrarli
    3. Salva il risultato in cache
    
    Args:
        user_id: ID dell'utente
        text (str): Testo da analizzare (post, narrativa base o trajectory)
        api_key (str): Chiave API OpenAI
        source_type (str): Tipo di sorgente ("posts", "narrative_base", "narrative_traj")
        
    Returns:
        tuple: (dizionario_topic, bool_da_cache)
               - dizionario con chiavi: positivetopics, neutraltopics, negativetopics
               - True se caricato dalla cache, False se generato da API
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Costruisce il percorso cache univoco: include user_id e tipo sorgente
    # Esempio: "2714_posts_topics.json" o "2714_narrative_base_topics.json"
    cache_file = os.path.join(CACHE_DIR, f"{user_id}_{source_type}_topics.json")
    
    # Fase 1: Controlla se i topic sono già in cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)  # Carica i topic dalla cache
            return data, True  # True indica che i dati vengono dalla cache
        except Exception as e:
            st.error(f"Error loading cache: {e}")  # Mostra errore se la cache è corrotta
    
    # Fase 2: Se non c'è cache, chiama l'API
    if not api_key:
        return None, False  # Senza API key non può procedere
    
    # Crea il client OpenAI
    client = OpenAI(api_key=api_key)
    
    # Costruisce il prompt per l'estrazione granulare dei topic
    prompt = build_prompt_granular(text)
    
    try:
        # Chiama l'API GPT con il prompt
        response = client.chat.completions.create(
            model="gpt-5.1",  # Modello da utilizzare
            messages=[
                {"role": "system", "content": "You are an expert assistant for topic extraction. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }  # Forza output JSON valido
        )
        
        # Estrae e decodifica il JSON dalla risposta
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Fase 3: Salva i topic estratti in cache per usi futuri
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        return data, False  # False indica che i dati sono freschi da API
        
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")  # Mostra errore nell'interfaccia
        return None, False
