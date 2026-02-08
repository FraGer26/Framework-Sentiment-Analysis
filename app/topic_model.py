import os
import json
import streamlit as st
from openai import OpenAI

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "topics")

def build_prompt_granular(text):
    prompt = f"""
Input
A text describing a user’s experiences and emotional states over time.

TEXT:
{text}

Task
Extract the specific, concrete, and distinct topics that characterize the user’s experience.
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
    return prompt.strip()

def extract_topics(user_id, text, api_key, source_type="posts"):
    """
    Estrae argomenti per un dato utente usando OpenAI API.
    Controlla prima la cache basandosi su source_type.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Il nome del file cache include il tipo sorgente per separare i risultati
    cache_file = os.path.join(CACHE_DIR, f"{user_id}_{source_type}_topics.json")
    
    # 1. Controlla Cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, True # True indica cached
        except Exception as e:
            st.error(f"Error loading cache: {e}")
            
    # 2. Chiama API se no cache
    if not api_key:
        return None, False
        
    client = OpenAI(api_key=api_key)
    
    prompt = build_prompt_granular(text)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are an expert assistant for topic extraction. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" } # Forza output JSON
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # 3. Salva in Cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
        return data, False # False indica fresco da API
        
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None, False
