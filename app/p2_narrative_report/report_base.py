# =============================================================================
# MODULO: report_base.py
# DESCRIZIONE: Genera il report narrativo "Base" per un utente tramite GPT.
#              Il report Base analizza i post come serie temporale e descrive
#              l'evoluzione emotiva dell'utente in un unico paragrafo narrativo,
#              SENZA utilizzare informazioni sulla segmentazione.
# =============================================================================

import pandas as pd  # Libreria per manipolazione dati tabulari
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from openai import OpenAI  # Client ufficiale per le API OpenAI

# Directory dove vengono salvati i report base in cache
BASE_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "reports", "base")


def get_base_report_path(user_id):
    """
    Costruisce il percorso del file cache per il report base di un utente.
    
    Args:
        user_id: ID dell'utente
        
    Returns:
        str: Percorso assoluto del file JSON del report
    """
    return os.path.join(BASE_REPORT_DIR, f"{user_id}.json")


def load_base_report(user_id):
    """
    Carica il report base dalla cache su disco.
    
    Args:
        user_id: ID dell'utente
        
    Returns:
        dict: Dizionario con il report, o None se non esiste o è corrotto
    """
    # Costruisce il percorso del file cache
    path = get_base_report_path(user_id)
    # Verifica se il file esiste
    if os.path.exists(path):
        try:
            # Legge e decodifica il file JSON
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None  # Restituisce None se il file è corrotto
    return None  # Restituisce None se il file non esiste


def build_prompt_base(posts, timestamps):
    """
    Costruisce il prompt per la generazione del report base.
    
    Il prompt chiede a GPT di analizzare i post come serie temporale
    e produrre una narrativa coerente sull'evoluzione emotiva dell'utente,
    senza segmentazione o fasi.
    
    Args:
        posts: Serie/Lista dei testi dei post
        timestamps: Serie/Lista delle date dei post
        
    Returns:
        str: Prompt formattato per GPT
    """
    # Template del prompt con variabili interpolate
    prompt = f"""
Posts ($p$): {posts}
Timestamps ($t$): {timestamps}

Task:

You are given a series of social media posts $p$ written by a single user, sorted by timestamp $t$.
Each post in $p$ is associated with a timestamp $t$.
Analyze the posts $p$ as a time series and describe the user's emotional state across the entire period.
Identify any posts that show signs of depressive-leaning emotional expressions and explain why those posts stand out in context.
Describe the period in which these emotions appear most strongly and how the emotional tone changes from beginning to end.
Use the content of the posts $p$ to create a clear, coherent, and integrative narrative about the evolution of the user's emotional state over time.

Output:

The output must be limited exclusively to a single integrative analytical narrative describing the user's emotional evolution.
No bullet points, lists, section headings, or meta-commentary should be included.

"""
    # Rimuove spazi bianchi iniziali e finali dal prompt
    return prompt.strip()


def ask_chatgpt(client, prompt):
    """
    Invia un prompt a GPT e restituisce la risposta.
    
    Args:
        client: Istanza del client OpenAI
        prompt (str): Prompt da inviare al modello
        
    Returns:
        str: Testo della risposta generata dal modello
    """
    # Chiama l'API OpenAI con il modello specificato
    response = client.chat.completions.create(
        model="gpt-5.1",  # Modello GPT da utilizzare
        messages=[{"role": "user", "content": prompt}]  # Messaggio utente con il prompt
    )
    # Estrae e restituisce il contenuto della prima risposta
    return response.choices[0].message.content


def ask_base(posts, timestamps, client):
    """
    Genera l'analisi base chiamando GPT con il prompt costruito.
    
    Questa funzione replica la logica della cella del notebook originale:
    prompt_base = build_prompt_base(posts, timestamps)
    output_base = ask_chatgpt(prompt_base)
    
    Args:
        posts: Serie/Lista dei testi dei post
        timestamps: Serie/Lista delle date dei post
        client: Istanza del client OpenAI
        
    Returns:
        str: Narrativa base generata da GPT
    """
    # Costruisce il prompt con i dati dell'utente
    prompt_base = build_prompt_base(posts, timestamps)
    # Invia il prompt a GPT e restituisce la risposta
    return ask_chatgpt(client, prompt_base)


def generate_base_report(user_id, user_df, api_key, output_txt_path=None):
    """
    Genera il report base completo per un utente.
    
    Flusso:
    1. Controlla se esiste un report in cache
    2. Se non esiste, chiama GPT per generarlo
    3. Salva il risultato in cache JSON
    4. Opzionalmente salva anche come file TXT
    
    Args:
        user_id: ID dell'utente
        user_df (pd.DataFrame): DataFrame con i dati dell'utente
        api_key (str): Chiave API OpenAI
        output_txt_path (str, optional): Percorso per salvare anche come TXT
        
    Returns:
        tuple: (dizionario_risultato, bool_da_cache)
               - dizionario con chiave 'base_analysis'
               - True se caricato dalla cache, False se generato
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(BASE_REPORT_DIR):
        os.makedirs(BASE_REPORT_DIR, exist_ok=True)
        
    # Fase 1: Controlla se il report è già in cache
    cached = load_base_report(user_id)
    if cached:
        return cached, True  # Restituisce dalla cache con flag True

    # Se non c'è API key, non può generare
    if not api_key:
        return None, False

    # Crea il client OpenAI con la chiave API fornita
    client = OpenAI(api_key=api_key)
    
    # Fase 2: Prepara i dati (replica le variabili del notebook originale)
    posts = user_df["Text"]  # Colonna con i testi dei post
    timestamps = user_df["Date"]  # Colonna con le date
    
    try:
        # Fase 3: Chiama GPT per generare l'analisi base
        base_analysis = ask_base(posts, timestamps, client)
        
        # Costruisce il dizionario risultato
        results = {
            'base_analysis': base_analysis  # La narrativa generata
        }
        
    except Exception as e:
        # In caso di errore API, salva il messaggio di errore
        results = {
            'base_analysis': f"Error generating base report: {e}"
        }

    # Fase 4: Salva in cache come file JSON
    with open(get_base_report_path(user_id), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Salvataggio opzionale come file TXT (replica il comportamento del notebook)
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(results['base_analysis'])  # Scrive solo il testo narrativo
            
    # Restituisce il risultato con flag False (non da cache)
    return results, False
