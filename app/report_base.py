import pandas as pd
import os
import json
from openai import OpenAI

# Directory cache per report base
BASE_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "reports", "base")

def get_base_report_path(user_id):
    return os.path.join(BASE_REPORT_DIR, f"{user_id}.json")

def load_base_report(user_id):
    path = get_base_report_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    return None

def build_prompt_base(posts, timestamps):
    prompt = f"""
Posts ($p$): {posts}
Timestamps ($t$): {timestamps}

Task:

You are given a series of social media posts $p$ written by a single user, sorted by timestamp $t$.
Each post in $p$ is associated with a timestamp $t$.
Analyze the posts $p$ as a time series and describe the user’s emotional state across the entire period.
Identify any posts that show signs of depressive-leaning emotional expressions and explain why those posts stand out in context.
Describe the period in which these emotions appear most strongly and how the emotional tone changes from beginning to end.
Use the content of the posts $p$ to create a clear, coherent, and integrative narrative about the evolution of the user’s emotional state over time.

Output:

The output must be limited exclusively to a single integrative analytical narrative describing the user’s emotional evolution.
No bullet points, lists, section headings, or meta-commentary should be included.

"""
    return prompt.strip()

def ask_chatgpt(client, prompt):
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def ask_base(posts, timestamps, client):
    """
    Mimica la cella di generazione base.
    """
    # Notebook:
    # prompt_base = build_prompt_base(posts, timestamps)
    # output_base=ask_chatgpt(prompt_base)
    
    prompt_base = build_prompt_base(posts, timestamps)
    return ask_chatgpt(client, prompt_base)

def generate_base_report(user_id, user_df, api_key, output_txt_path=None):
    """
    Genera il Report Base. Salva in cache JSON e opzionalmente TXT.
    """
    if not os.path.exists(BASE_REPORT_DIR):
        os.makedirs(BASE_REPORT_DIR, exist_ok=True)
        
    # 1. Controlla Cache
    cached = load_base_report(user_id)
    if cached:
        return cached, True

    if not api_key:
        return None, False

    client = OpenAI(api_key=api_key)
    
    # 2. Prepara Dati (corrispondenza variabili notebook)
    posts = user_df["Text"]
    timestamps = user_df["Date"]
    
    try:
        # 3. Chiedi base (Mimica cella 227)
        # Dobbiamo gestire potenziali errori, anche se il notebook non mostra blocchi espliciti per questa cella.
        base_analysis = ask_base(posts, timestamps, client)
        
        results = {
            'base_analysis': base_analysis
        }
        
    except Exception as e:
        results = {
            'base_analysis': f"Error generating base report: {e}"
        }

    # Salva in Cache JSON
    with open(get_base_report_path(user_id), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Salvataggio TXT opzionale (Notebook salva semplicemente come file di testo)
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(results['base_analysis'])
            
    return results, False
