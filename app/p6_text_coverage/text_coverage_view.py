# =============================================================================
# MODULO: text_coverage_view.py
# DESCRIZIONE: Renderizza la sezione di copertura testuale nell'interfaccia.
#              Confronta i post grezzi dell'utente con il testo della narrativa
#              generata, mostrando quanto la narrativa "copre" il contenuto
#              dei post originali in termini di similarità semantica.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import plotly.graph_objects as go  # Libreria per grafici interattivi
from p2_narrative_report import report_base, report_trajectory  # Moduli per caricare i report
from p6_text_coverage import text_coverage  # Modulo per il calcolo della copertura testuale


def render_text_coverage(selected_user, user_data):
    """
    Renderizza la sezione di copertura testuale.
    
    Confronta i post grezzi dell'utente con la narrativa selezionata
    e mostra:
    1. Metriche: Precision, Recall (Coverage), F1
    2. Conteggio post coperti
    3. Grafico di sensibilità alla soglia
    4. Tabella delle corrispondenze
    
    Args:
        selected_user: ID dell'utente selezionato
        user_data (pd.DataFrame): DataFrame con i dati dell'utente
    """
    
    # Converte tutti i post in una lista di stringhe (testi completi)
    texts_full = user_data["Text"].astype(str).tolist()
    
    # ==========================================================================
    # CARICAMENTO NARRATIVE DALLA CACHE
    # ==========================================================================
    base_data = report_base.load_base_report(selected_user)  # Carica report base
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)  # Carica report trajectory
    
    narrative_base = ""  # Testo della narrativa base
    narrative_traj = ""  # Testo della narrativa trajectory
    
    # Estrae i testi dalle strutture dati caricate
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    if traj_data_loaded:
        narrative_traj = traj_data_loaded.get('trajectory_summary', '')

    # ==========================================================================
    # SELEZIONE SORGENTE E PARAMETRI
    # ==========================================================================
    # Selettore per scegliere quale narrativa confrontare
    coverage_source = st.selectbox(
        "Compare Against",
        ["Narrative Base", "Narrative Trajectory"],
        key="cov_source"  # Chiave univoca per lo stato Streamlit
    )
    # Seleziona il testo appropriato
    comparison_text = narrative_base if coverage_source == "Narrative Base" else narrative_traj
    
    # Avviso se il testo selezionato è vuoto
    if not comparison_text:
        st.warning("Selected narrative text is empty.")
    
    # Slider per regolare la soglia di similarità
    coverage_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # ==========================================================================
    # CALCOLO O CARICAMENTO DELLA COPERTURA
    # ==========================================================================
    metrics = None  # Metriche di copertura
    df_matches = None  # DataFrame delle corrispondenze
    
    # Determina la chiave per la cache
    source_key = "base" if coverage_source == "Narrative Base" else "traj"
     
    # Controlla se i risultati sono in cache
    is_cached = text_coverage.check_cache(selected_user, source_key, threshold=coverage_threshold)
    
    # Caricamento automatico dalla cache se disponibile
    if is_cached and 'coverage_metrics' not in st.session_state:
        metrics, df_matches, _ = text_coverage.load_cache(selected_user, source_key, threshold=coverage_threshold)
        if metrics:
            # Salva nello stato sessione per persistenza
            st.session_state['coverage_metrics'] = metrics
            st.session_state['coverage_matches'] = df_matches
            st.success("Coverage results loaded from cache.")
    
    # Pulsante per calcolare o ricalcolare la copertura
    if st.button("Calculate Coverage") or (is_cached and 'coverage_metrics' not in st.session_state):
        # Calcola solo se non già caricato e il testo di confronto è disponibile
        if not metrics and comparison_text:
            with st.spinner("Calculating semantic coverage..."):
                # Calcola le metriche di copertura testuale
                metrics, df_matches, _ = text_coverage.calculate_text_coverage_metrics(
                    selected_user, source_key, texts_full, comparison_text, threshold=coverage_threshold
                )
                # Salva nello stato sessione
                st.session_state['coverage_metrics'] = metrics
                st.session_state['coverage_matches'] = df_matches

    # ==========================================================================
    # VISUALIZZAZIONE RISULTATI
    # ==========================================================================
    if 'coverage_metrics' in st.session_state:
        res = st.session_state['coverage_metrics']  # Metriche
        df = st.session_state['coverage_matches']  # DataFrame corrispondenze
        
        # Mostra le 3 metriche principali in colonne affiancate
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{res['precision']:.2f}")  # Precisione
        c2.metric("Recall (Coverage)", f"{res['recall']:.2f}")  # Copertura
        c3.metric("F1 Score", f"{res['f1']:.2f}")  # Score bilanciato
        
        # Mostra il conteggio dei post coperti
        st.write(f"**{res['tp_recall']}** / **{res['n_full']}** posts covered.")
        
        # --- Grafico Analisi di Sensibilità alla Soglia ---
        sens_data = text_coverage.sensitivity_analysis_text(texts_full, comparison_text)
        if sens_data:
            fig_sens = go.Figure()
            # Linea F1 con marcatori
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
            # Linea Precision tratteggiata
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
            # Linea Recall tratteggiata
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
            
            # Linea verticale rossa alla soglia selezionata
            fig_sens.add_vline(x=coverage_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
            # Configurazione layout
            fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.error("Could not generate sensitivity data.")
        
        # --- Tabella delle Corrispondenze ---
        if df is not None and not df.empty:
            st.write("#### Matches")
            # Mostra solo i post coperti (covered == True)
            st.dataframe(df[df['covered'] == True], use_container_width=True)
        else:
            st.warning("No detail data available for matches.")
