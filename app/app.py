# =============================================================================
# MODULO: app.py
# DESCRIZIONE: Punto di ingresso principale dell'applicazione Streamlit.
#              Orchestrare tutte le fasi dell'analisi comportamentale:
#              - Caricamento dati e selezione utente
#              - Calcolo rischio EMA e segmentazione
#              - Navigazione tra le 8 sezioni di analisi
#              - Gestione della chiave API OpenAI
#              Ogni sezione delega la logica al modulo view corrispondente.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import pandas as pd  # Libreria per manipolazione dati tabulari
import os  # Libreria per operazioni su file e percorsi

# Import dei moduli per ogni fase dell'analisi
from p0_global import data, dataset_stats_view, overview_view  # Dati e viste globali
from p1_segmentation import ema, segment, risk_view  # Rischio EMA e segmentazione
from p2_narrative_report import trajectory_view, report_base, report_trajectory  # Report narrativi
from p3_llm_judge import gpt_evaluation_view  # Valutazione GPT cieca
from p4_topic_analysis import topic_analysis_view  # Estrazione topic
from p5_topic_coverage import topic_coverage_view  # Copertura topic
from p6_text_coverage import text_coverage_view  # Copertura testuale
from p7_topic_analysis_clustering import clustering_view  # Clustering avanzato

# =============================================================================
# CONFIGURAZIONE PAGINA STREAMLIT
# Imposta il titolo della pagina, il layout wide e la sidebar espansa
# =============================================================================
st.set_page_config(
    page_title="Analisi Utenti Reddit",  # Titolo nella tab del browser
    layout="wide",  # Layout a tutta larghezza
    initial_sidebar_state="expanded"  # Sidebar aperta di default
)


def main():
    """
    Funzione principale dell'applicazione.
    
    Gestisce l'intero flusso:
    1. Configurazione API key e caricamento dati
    2. Selezione tra analisi singolo utente e statistiche globali
    3. Per analisi singolo utente: calcolo rischio, segmentazione,
       metriche intestazione e navigazione tra le 8 sezioni
    """
    # Titolo e sottotitolo della dashboard
    st.title("🧠 Dashboard Analisi Comportamentale")
    st.markdown("### Valutazione Rischio e Segmentazione Utenti Reddit")
    
    # ==========================================================================
    # BARRA LATERALE: DATI & CONTROLLO
    # ==========================================================================
    st.sidebar.header("Dati & Controllo")
    
    # --- Gestione Chiave API OpenAI ---
    # L'utente può inserire manualmente la chiave API o usare i segreti Streamlit
    manual_api_key = st.sidebar.text_input(
        "Chiave API OpenAI",
        type="password",  # Maschera l'input
        help="Richiesto per estrarre argomenti o generare sommari qualitativi."
    )
    
    # Logica di priorità: 1. Input manuale, 2. Segreti Streamlit
    try:
        if manual_api_key:
            api_key = manual_api_key  # L'input manuale ha priorità
        elif "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]  # Fallback ai segreti
        else:
            api_key = ""  # Nessuna chiave disponibile
    except:
        api_key = manual_api_key if manual_api_key else ""
    
    # ==========================================================================
    # CARICAMENTO DATI
    # Cerca automaticamente il file CSV nella directory classification
    # ==========================================================================
    # Calcola i percorsi relativi alla posizione dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    rel_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    
    uploaded_file = None  # Placeholder per upload manuale (non implementato)
    
    # Tenta di caricare il dataset in ordine di priorità
    df = None
    if uploaded_file:
        df = data.load_data(uploaded_file)  # Da file caricato manualmente
    elif os.path.exists(abs_path):
        df = data.load_data(abs_path)  # Percorso assoluto
    elif os.path.exists(rel_path):
        df = data.load_data(rel_path)  # Percorso relativo
    else:
        # Nessun file trovato: mostra i percorsi cercati e interrompe
        st.sidebar.warning(f"File predefinito non trovato. Controllato:\\n- `{abs_path}`\\n- `{rel_path}`\\nCaricare manualmente.")
        st.stop()
    
    # Verifica che il caricamento sia riuscito
    if df is None:
        st.error("Fallimento nel caricamento dati.")
        st.stop()
    
    # ==========================================================================
    # SELEZIONE MODALITÀ DI ANALISI
    # Due modalità: analisi per singolo utente o statistiche globali del dataset
    # ==========================================================================
    analysis_mode = st.sidebar.radio(
        "Livello Analisi",
        ["👤 Analisi Singolo Utente", "🌍 Statistiche Globali Dataset"]
    )
    st.sidebar.markdown("---")

    # ==========================================================================
    # MODALITÀ: STATISTICHE GLOBALI DATASET
    # Mostra statistiche aggregate calcolate con Spark SQL
    # ==========================================================================
    if analysis_mode == "🌍 Statistiche Globali Dataset":
        dataset_stats_view.render_dataset_statistics(df, api_key)
        
    # ==========================================================================
    # MODALITÀ: ANALISI SINGOLO UTENTE
    # Permette di selezionare un utente e navigare tra le sezioni di analisi
    # ==========================================================================
    else:
        # --- Selezione Utente ---
        users = list(data.get_subject_ids(df))  # Lista degli ID utente
        
        # Imposta l'utente 2714 come predefinito (se esiste)
        default_index = 0
        target_user = 2714
        
        # Gestisce possibili disallineamenti di tipo (int vs str)
        if target_user in users:
            default_index = users.index(target_user)
        elif str(target_user) in users:
            default_index = users.index(str(target_user))
        
        # Selettore utente nella sidebar
        selected_user = st.sidebar.selectbox(
            "Seleziona Utente (Subject ID)",
            users,
            index=default_index
        )
    
        # --- Parametri Globali ---
        st.sidebar.markdown("### ⚙️ Parametri Analisi")
        # Slider per il decadimento EMA (influenza la sensibilità del rischio)
        half_life = st.sidebar.slider(
            "Decadimento EMA (Giorni)", 1, 60, 15,
            help="Controlla il decadimento del Punteggio di Rischio."
        )
        # Slider per il numero di segmenti della traiettoria
        k_segments = st.sidebar.slider(
            "K-Segmenti", 1, 20, 10,
            help="Numero di segmenti per la traiettoria."
        )
    
        # --- Filtraggio Dati Utente ---
        user_data = data.get_user_data(df, selected_user)  # Filtra il DataFrame per l'utente
        
        if user_data.empty:
            st.warning("Nessun dato per l'utente selezionato.")
            st.stop()
    
        # =================================================================
        # CALCOLI GLOBALI
        # Eseguiti una volta per l'utente selezionato
        # =================================================================
        # Calcola il punteggio di rischio EMA
        risk_series = ema.calculate_risk_score(user_data, half_life)
        # Segmenta la serie temporale del rischio
        if not risk_series.empty:
            segments = segment.segment_time_series(risk_series, k_segments)
        else:
            segments = []
            
        # =================================================================
        # METRICHE INTESTAZIONE
        # Mostra le statistiche chiave dell'utente in 5 colonne
        # =================================================================
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Colonna 1: ID utente e totale post
        col1.metric("ID Utente", str(selected_user))
        col1.metric("Totale Post", len(user_data))
        
        # Colonna 2: Intervallo temporale dei post
        start_date = user_data["Date"].min().strftime("%Y-%m-%d")
        end_date = user_data["Date"].max().strftime("%Y-%m-%d")
        col2.metric("Primo Post", start_date)
        col2.metric("Ultimo Post", end_date)
        
        # Colonna 3-5: Metriche di rischio (inizialmente placeholder)
        avg_score_metric = col3.empty()
        peak_score_metric = col4.empty()
        current_risk_metric = col5.empty()
        
        # Calcola le metriche dal punteggio di rischio EMA
        avg_score = risk_series.mean() if not risk_series.empty else 0  # Media
        peak_score = risk_series.max() if not risk_series.empty else 0  # Picco
        current_score = risk_series.iloc[-1] if not risk_series.empty else 0  # Corrente
        
        # Popola le metriche nei placeholder
        avg_score_metric.metric("Punteggio Rischio Medio", f"{avg_score:.4f}")
        peak_score_metric.metric("Picco Punteggio Rischio", f"{peak_score:.4f}")
        # Il delta mostra la differenza tra rischio corrente e medio
        current_risk_metric.metric(
            "Rischio Corrente", f"{current_score:.4f}",
            delta=f"{current_score - avg_score:.4f}"
        )
        
        st.markdown("---")  # Separatore visivo
    
        # =================================================================
        # NAVIGAZIONE SEZIONI DI ANALISI
        # 8 sezioni, ognuna delegata al modulo view corrispondente
        # =================================================================
        analysis_section = st.sidebar.radio(
            "Sezione Analisi Utente",
            [
                "👤 Panoramica Utente",  # p0: Overview
                "📊 Dashboard Rischio",  # p1: EMA + Segmentazione
                "📖 Traiettoria Narrativa",  # p2: Report GPT
                "⚖️ Valutazione GPT",  # p3: LLM Judge A/B test
                "🧩 Analisi Argomenti",  # p4: Topic extraction
                "🧩 Copertura Argomenti",  # p5: Topic coverage
                "📄 Copertura Testo",  # p6: Text coverage
                "🔍 Clustering Argomenti"  # p7: UMAP + HDBSCAN clustering
            ]
        )
        
        # Mostra il titolo della sezione selezionata
        st.subheader(f"{analysis_section}")
        st.caption(f"Utente Selezionato: {selected_user}")
    
        # =================================================================
        # ROUTING DELLE SEZIONI
        # Ogni sezione chiama la funzione render del modulo corrispondente
        # =================================================================
        if analysis_section == "👤 Panoramica Utente":
            # Mostra timeline attività, post recenti e post a rischio
            overview_view.render_global_overview(user_data)
            
        elif analysis_section == "📊 Dashboard Rischio":
            # Mostra grafico EMA colorato per segmento e tabella breakpoint
            risk_view.render_risk_dashboard(user_data, risk_series, segments, half_life)
        
        elif analysis_section == "📖 Traiettoria Narrativa":
            # Mostra/genera report narrativi Base e Trajectory con GPT
            trajectory_view.render_trajectory_section(selected_user, user_data, segments, api_key)
    
        elif analysis_section == "⚖️ Valutazione GPT":
            # Esegue il test cieco A/B tra report Base e Trajectory
            gpt_evaluation_view.render_gpt_evaluation(selected_user, api_key)
    
        elif analysis_section == "🧩 Analisi Argomenti":
            # Estrae topic dai post o dalle narrative con GPT
            topic_analysis_view.render_topic_analysis(selected_user, user_data, api_key)
    
        elif analysis_section == "🧩 Copertura Argomenti":
            # Confronta topic ground truth con topic dalla narrativa
            topic_coverage_view.render_topic_coverage(selected_user)
    
        elif analysis_section == "📄 Copertura Testo":
            # Confronta post grezzi con narrativa usando embedding
            text_coverage_view.render_text_coverage(selected_user, user_data)
    
        elif analysis_section == "🔍 Clustering Argomenti":
            # Clustering avanzato con UMAP + HDBSCAN
            clustering_view.render_clustering_section(selected_user, user_data)


# =============================================================================
# ENTRY POINT
# Avvia l'applicazione quando il file viene eseguito direttamente
# =============================================================================
if __name__ == "__main__":
    main()
