# =============================================================================
# MODULO: topic_analysis_view.py
# DESCRIZIONE: Renderizza la sezione di analisi dei topic nell'interfaccia Streamlit.
#              Permette di scegliere la sorgente del testo (Post, Narrativa Base,
#              Narrativa Trajectory) e mostra i topic organizzati per valenza
#              emotiva (positivi, neutri, negativi).
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from p2_narrative_report import report_base, report_trajectory  # Moduli per caricare le narrative
from p4_topic_analysis import topic_model  # Modulo per l'estrazione dei topic


def render_topic_analysis(selected_user, user_data, api_key):
    """
    Renderizza la sezione completa dell'analisi dei topic.
    
    Permette all'utente di:
    1. Scegliere la sorgente del testo da analizzare
    2. Caricare topic dalla cache o generarli con GPT
    3. Visualizzare i topic organizzati per valenza emotiva
    
    Args:
        selected_user: ID dell'utente selezionato
        user_data (pd.DataFrame): DataFrame con i dati dell'utente
        api_key (str): Chiave API OpenAI
    """
    
    # ==========================================================================
    # FASE 1: SELEZIONE SORGENTE
    # L'utente sceglie quale testo analizzare per l'estrazione dei topic
    # ==========================================================================
    topic_source = st.selectbox(
        "Select Topic Source",  # Etichetta del selettore
        ["Raw Posts", "Narrative Base", "Narrative Trajectory"],  # Opzioni disponibili
        key="topic_source_select"  # Chiave univoca per lo stato Streamlit
    )
    
    # Inizializza le variabili per il testo e la chiave sorgente
    text_to_analyze = ""  # Testo che verrà analizzato
    source_key = "posts"  # Chiave usata per il nome del file cache (default: posts)
    
    # ==========================================================================
    # CARICAMENTO TESTO IN BASE ALLA SORGENTE SELEZIONATA
    # ==========================================================================
    if topic_source == "Raw Posts":
        # Concatena tutti i post dell'utente in un unico testo
        text_to_analyze = "\n".join(user_data["Text"].astype(str))
        source_key = "posts"
    else:
        # Carica le narrative dalla cache per le opzioni Base e Trajectory
        base_data = report_base.load_base_report(selected_user)
        traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
        
        narrative_base = ""  # Testo della narrativa base
        narrative_traj = ""  # Testo della narrativa trajectory
        
        # Estrae la narrativa base dal dizionario
        if base_data:
            narrative_base = base_data.get('base_analysis', '')
        # Estrae il riepilogo della traiettoria
        if traj_data_loaded:
            narrative_traj = traj_data_loaded.get('trajectory_summary', '')
        
        # Seleziona il testo appropriato in base alla sorgente scelta
        if topic_source == "Narrative Base":
            text_to_analyze = narrative_base
            source_key = "narrative_base"  # Chiave per il file cache
        elif topic_source == "Narrative Trajectory":
            text_to_analyze = narrative_traj
            source_key = "narrative_traj"  # Chiave per il file cache
    
    # Se non c'è testo disponibile, mostra un avviso e interrompe
    if not text_to_analyze:
        st.warning(f"No text available for {topic_source}. Please run Narrative Analysis or check data.")
        return

    # ==========================================================================
    # FASE 2: CARICAMENTO O GENERAZIONE TOPIC
    # Controlla la cache per topic già estratti, altrimenti offre la possibilità di generarli
    # ==========================================================================
    
    # Costruisce il percorso cache per questa combinazione utente/sorgente
    cache_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{source_key}_topics.json")
    is_cached = os.path.exists(cache_path)  # Verifica se la cache esiste
    
    topics_data = None  # Inizializza i dati dei topic
    
    if is_cached:
        # I topic sono in cache: li carica direttamente
        st.success(f"Topic analysis for **{topic_source}** available in cache.")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)  # Carica i topic dalla cache
        except:
            pass  # Ignora errori di lettura
    else:
        # I topic non sono in cache: mostra pulsante per generarli
        st.info(f"No cached topics for **{topic_source}**.")
        if st.button(f"Extract Topics ({topic_source})"):
            # Verifica che la chiave API sia disponibile
            if not api_key:
                st.error("API Key required.")
            else:
                with st.spinner("Extracting topics..."):
                    # Chiama il modulo topic_model per estrarre i topic con GPT
                    topics_data, _ = topic_model.extract_topics(
                        selected_user,  # ID utente
                        text_to_analyze,  # Testo da analizzare
                        api_key,  # Chiave API OpenAI
                        source_type=source_key  # Tipo di sorgente per la cache
                    )
                    if topics_data:
                        st.success("Topics extracted!")
                        st.rerun()  # Ricarica la pagina per mostrare i risultati
    
    # ==========================================================================
    # FASE 3: VISUALIZZAZIONE TOPIC
    # Mostra i topic organizzati in 3 colonne per valenza emotiva
    # ==========================================================================
    if topics_data:
        # Crea 3 colonne affiancate per le tre categorie
        col_list1, col_list2, col_list3 = st.columns(3)
        
        # Colonna sinistra: Topic Positivi
        with col_list1:
            st.markdown("#### Positive Topics")
            # Itera sulla lista dei topic positivi e li mostra come elenco puntato
            for t in topics_data.get("positivetopics", []): st.write(f"- {t}")
        
        # Colonna centrale: Topic Neutri
        with col_list2:
            st.markdown("#### Neutral Topics")
            for t in topics_data.get("neutraltopics", []): st.write(f"- {t}")
        
        # Colonna destra: Topic Negativi
        with col_list3:
            st.markdown("#### Negative Topics")
            for t in topics_data.get("negativetopics", []): st.write(f"- {t}")
