# =============================================================================
# MODULO: topic_coverage_view.py
# DESCRIZIONE: Renderizza la sezione di copertura dei topic nell'interfaccia.
#              Confronta i topic estratti dai post grezzi (ground truth) con
#              quelli estratti dalle narrative, mostrando metriche e grafici
#              di sensibilità alla soglia di similarità.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from p5_topic_coverage import topic_coverage  # Modulo per il calcolo della copertura
from p4_topic_analysis import topic_model  # Modulo per accedere alla cache dei topic


def render_topic_coverage(selected_user):
    """
    Renderizza la sezione di copertura dei topic.
    
    Confronta i topic di riferimento (dai post grezzi) con quelli candidati
    (dalla narrativa Base o Trajectory) e mostra:
    1. Metriche: Precision, Recall, F1
    2. Grafico di sensibilità alla soglia
    3. Tabella delle corrispondenze trovate
    
    Args:
        selected_user: ID dell'utente selezionato
    """

    st.caption("Calculate how well the Narrative Topics cover the Raw Ground Truth Topics.")
    
    # ==========================================================================
    # FASE 1: CARICAMENTO TOPIC DI RIFERIMENTO (GROUND TRUTH)
    # I topic estratti dai post grezzi fungono da riferimento
    # ==========================================================================
    raw_topics = []  # Lista dei topic di riferimento
    # Costruisce il percorso del file cache dei topic dei post grezzi
    rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
    
    if os.path.exists(rt_path):
        try:
            with open(rt_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                # Combina tutti i topic (positivi + neutri + negativi) in un'unica lista
                raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
    
    # Se non ci sono topic di riferimento, mostra avviso e interrompe
    if not raw_topics:
        st.warning("Ground Truth (Raw Posts) topics not found. Please extract them in 'Topic Analysis' first.")
        return

    # ==========================================================================
    # FASE 2: SELEZIONE E CARICAMENTO TOPIC CANDIDATI
    # L'utente sceglie quale narrativa usare come candidata
    # ==========================================================================
    candidate_source = st.selectbox(
        "Select Candidate Topics",
        ["Narrative Base", "Narrative Trajectory"],
        key="topic_cov_cand"  # Chiave univoca per lo stato Streamlit
    )
    # Determina la chiave cache in base alla sorgente scelta
    cand_key = "narrative_base" if candidate_source == "Narrative Base" else "narrative_traj"
    
    # Carica i topic candidati dalla cache
    cand_topics = []  # Lista dei topic candidati
    ct_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{cand_key}_topics.json")
    if os.path.exists(ct_path):
        try:
            with open(ct_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                # Combina tutti i topic candidati in un'unica lista
                cand_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
    
    # Se non ci sono topic candidati, mostra avviso e interrompe
    if not cand_topics:
        st.warning(f"Candidate topics ({candidate_source}) not found. Please extract them in 'Topic Analysis' first.")
        return
    
    # Mostra il conteggio dei topic disponibili
    st.write(f"**Reference:** {len(raw_topics)} topics | **Candidate:** {len(cand_topics)} topics")
    
    # Slider per regolare la soglia di similarità coseno
    cov_threshold = st.slider("Topic Similarity Threshold", 0.5, 1.0, 0.75, 0.05)
    
    # ==========================================================================
    # FASE 3: CALCOLO O CARICAMENTO DELLA COPERTURA
    # ==========================================================================
    
    # Controlla se i risultati sono in cache per questa configurazione
    is_cached = topic_coverage.check_cache(selected_user, cand_key, threshold=cov_threshold)
    metrics = None  # Metriche di copertura
    df_matches = None  # DataFrame delle corrispondenze

    # Caricamento automatico dalla cache
    if is_cached and 'topic_metrics' not in st.session_state:
        metrics, df_matches, _ = topic_coverage.load_cache(selected_user, cand_key, threshold=cov_threshold)
        if metrics:
            st.success("Topic coverage results loaded from cache.")

    # Pulsante per calcolare o ricalcolare, oppure auto-calcolo se cache disponibile
    if st.button("Calculate Topic Coverage") or (is_cached and metrics is None):
        if not metrics:
            with st.spinner("Calculating matching..."):
                # Calcola le metriche di copertura usando embedding e similarità coseno
                metrics, df_matches, _ = topic_coverage.calculate_coverage_metrics(
                    selected_user, cand_key, raw_topics, cand_topics, threshold=cov_threshold
                )
    
    # ==========================================================================
    # FASE 4: VISUALIZZAZIONE RISULTATI
    # ==========================================================================
    if metrics:
        # Mostra le 3 metriche principali in colonne affiancate
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{metrics['precision']:.2f}")  # Precisione
        c2.metric("Recall", f"{metrics['recall']:.2f}")  # Copertura
        c3.metric("F1 Score", f"{metrics['f1']:.2f}")  # Score bilanciato
        
        # --- Grafico Analisi di Sensibilità alla Soglia ---
        sens_data = topic_coverage.sensitivity_analysis(raw_topics, cand_topics)
        
        if sens_data:
            import plotly.graph_objects as go  # Import locale per Plotly
            fig_sens = go.Figure()
            # Linea F1 con marcatori
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
            # Linea Precision tratteggiata
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
            # Linea Recall tratteggiata
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
            
            # Linea verticale rossa alla soglia selezionata
            fig_sens.add_vline(x=cov_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
            
            # Configurazione layout del grafico
            fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.error("Could not generate sensitivity data.")

        # --- Tabella delle Corrispondenze ---
        st.write("#### Matches")
        # Mostra solo le corrispondenze che superano la soglia
        st.dataframe(df_matches[df_matches['matched'] == True], use_container_width=True)
