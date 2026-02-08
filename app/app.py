import streamlit as st
import pandas as pd
import os
from p0_global import data, dataset_stats_view, overview_view
from p1_segmentation import ema, segment, risk_view
from p2_narrative_report import trajectory_view, report_base, report_trajectory
from p3_llm_judge import gpt_evaluation_view
from p4_topic_analysis import topic_analysis_view
from p5_topic_coverage import topic_coverage_view
from p6_text_coverage import text_coverage_view
from p7_topic_analysis_clustering import clustering_view

# --- Configurazione ---
st.set_page_config(
    page_title="Analisi Utenti Reddit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Interfaccia Utente Principale ---

def main():
    st.title("🧠 Dashboard Analisi Comportamentale")
    st.markdown("### Valutazione Rischio e Segmentazione Utenti Reddit")
    
    # -- Barra Laterale --
    st.sidebar.header("Dati & Controllo")
    
    # Chiave API per Argomenti/Valutazioni
    # Priorità: 1. Input Manuale, 2. Segreti Streamlit
    manual_api_key = st.sidebar.text_input("Chiave API OpenAI", type="password", help="Richiesto per estrarre argomenti o generare sommari qualitativi.")
    
    # Usa i segreti se disponibili come fallback
    try:
        if manual_api_key:
            api_key = manual_api_key
        elif "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = ""
    except:
        api_key = manual_api_key if manual_api_key else ""

    
    # Ricerca Automatica File
    # Percorsi consolidati nella sottocartella app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    rel_path = os.path.join(script_dir, "..", "classification", "output_Classification.csv")
    
    uploaded_file = None
    
    df = None
    if uploaded_file:
        df = data.load_data(uploaded_file)
    elif os.path.exists(abs_path):
        df = data.load_data(abs_path)
    elif os.path.exists(rel_path):
        df = data.load_data(rel_path)
    else:
        st.sidebar.warning(f"File predefinito non trovato. Controllato:\\n- `{abs_path}`\\n- `{rel_path}`\\nCaricare manualmente.")
        st.stop()
        
    if df is None:
        st.error("Fallimento nel caricamento dati.")
        st.stop()
        
    analysis_mode = st.sidebar.radio("Livello Analisi", ["👤 Analisi Singolo Utente", "🌍 Statistiche Globali Dataset"])
    st.sidebar.markdown("---")

    if analysis_mode == "🌍 Statistiche Globali Dataset":
        dataset_stats_view.render_dataset_statistics(df, api_key)
        
    else: # Analisi Utente Singolo
        # Selezione Utente (in cache)
        users = list(data.get_subject_ids(df))
        
        # Indice predefinito per utente 2714
        default_index = 0
        target_user = 2714
        
        # Gestione disallineamento int/str
        if target_user in users:
            default_index = users.index(target_user)
        elif str(target_user) in users:
            default_index = users.index(str(target_user))
            
        selected_user = st.sidebar.selectbox("Seleziona Utente (Subject ID)", users, index=default_index)
            

    
        # -- Parametri Globali --
        st.sidebar.markdown("### ⚙️ Parametri Analisi")
        half_life = st.sidebar.slider("Decadimento EMA (Giorni)", 1, 60, 15, help="Controlla il decadimento del Punteggio di Rischio.")
        k_segments = st.sidebar.slider("K-Segmenti", 1, 20, 10, help="Numero di segmenti per la traiettoria.")
    
        # -- Filtro Dati Utente (in cache) --
        user_data = data.get_user_data(df, selected_user)
        
        if user_data.empty:
            st.warning("Nessun dato per l'utente selezionato.")
            st.stop()
    
        # -- Calcoli Globali --
        # Calcola Punteggio di Rischio e Segmenti una volta per l'utente
        risk_series = ema.calculate_risk_score(user_data, half_life)
        if not risk_series.empty:
             segments = segment.segment_time_series(risk_series, k_segments)
        else:
             segments = []
            
        # -- Statistiche Intestazione --
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("ID Utente", str(selected_user))
        col1.metric("Totale Post", len(user_data))
        
        start_date = user_data["Date"].min().strftime("%Y-%m-%d")
        end_date = user_data["Date"].max().strftime("%Y-%m-%d")
        col2.metric("Primo Post", start_date)
        col2.metric("Ultimo Post", end_date)
        
        # Placeholder per Metriche di Rischio
        avg_score_metric = col3.empty()
        peak_score_metric = col4.empty()
        current_risk_metric = col5.empty()
        
        # Calcolo metriche per intestazione
        avg_score = risk_series.mean() if not risk_series.empty else 0
        peak_score = risk_series.max() if not risk_series.empty else 0
        current_score = risk_series.iloc[-1] if not risk_series.empty else 0
        
        avg_score_metric.metric("Punteggio Rischio Medio", f"{avg_score:.4f}")
        peak_score_metric.metric("Picco Punteggio Rischio", f"{peak_score:.4f}")
        current_risk_metric.metric("Rischio Corrente", f"{current_score:.4f}", delta=f"{current_score - avg_score:.4f}")
        
        st.markdown("---")
    
        # -- Navigazione Barra Laterale --
        
        analysis_section = st.sidebar.radio(
            "Sezione Analisi Utente", 
            [
                "👤 Panoramica Utente",
                "📊 Dashboard Rischio", 
                "📖 Traiettoria Narrativa", 
                "⚖️ Valutazione GPT",
                "🧩 Analisi Argomenti", 
                "🧩 Copertura Argomenti",
                "📄 Copertura Testo",
                "🔍 Clustering Argomenti"
            ]
        )
        
        st.subheader(f"{analysis_section}")
        st.caption(f"Utente Selezionato: {selected_user}")
    
        
        if analysis_section == "👤 Panoramica Utente":
            overview_view.render_global_overview(user_data)
            
        elif analysis_section == "📊 Dashboard Rischio":
            # Passa dati pre-calcolati
            risk_view.render_risk_dashboard(user_data, risk_series, segments, half_life)
        
        elif analysis_section == "📖 Traiettoria Narrativa":
            trajectory_view.render_trajectory_section(selected_user, user_data, segments, api_key)
    
        elif analysis_section == "⚖️ Valutazione GPT":
            gpt_evaluation_view.render_gpt_evaluation(selected_user, api_key)
    
        elif analysis_section == "🧩 Analisi Argomenti":
            topic_analysis_view.render_topic_analysis(selected_user, user_data, api_key)
    
        elif analysis_section == "🧩 Copertura Argomenti":
            topic_coverage_view.render_topic_coverage(selected_user)
    
        elif analysis_section == "📄 Copertura Testo":
            text_coverage_view.render_text_coverage(selected_user, user_data)
    
        elif analysis_section == "🔍 Clustering Argomenti":
            clustering_view.render_clustering_section(selected_user, user_data)
            

if __name__ == "__main__":
    main()
