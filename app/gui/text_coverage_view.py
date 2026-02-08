import streamlit as st
import report_base
import report_trajectory
import text_coverage
import plotly.graph_objects as go

def render_text_coverage(selected_user, user_data):
    
    texts_full = user_data["Text"].astype(str).tolist()
    
    base_data = report_base.load_base_report(selected_user)
    traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
    
    narrative_base = ""
    narrative_traj = ""
    
    if base_data:
        narrative_base = base_data.get('base_analysis', '')
    if traj_data_loaded:
        narrative_traj = traj_data_loaded.get('trajectory_summary', '')

    coverage_source = st.selectbox("Compare Against", ["Narrative Base", "Narrative Trajectory"], key="cov_source")
    comparison_text = narrative_base if coverage_source == "Narrative Base" else narrative_traj
        
    if not comparison_text:
        st.warning("Selected narrative text is empty.")
    
    coverage_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # Determina chiave tipo sorgente per cache
    metrics = None
    df_matches = None
    
    source_key = "base" if coverage_source == "Narrative Base" else "traj"
     
    # Controlla Cache
    is_cached = text_coverage.check_cache(selected_user, source_key, threshold=coverage_threshold)
    
    if is_cached and 'coverage_metrics' not in st.session_state:
        metrics, df_matches, _ = text_coverage.load_cache(selected_user, source_key, threshold=coverage_threshold)
        if metrics:
            st.session_state['coverage_metrics'] = metrics
            st.session_state['coverage_matches'] = df_matches
            st.success("Coverage results loaded from cache.")
            
    # Auto-avvio se logica fallisce o solo basato su stato
    if st.button("Calculate Coverage") or (is_cached and 'coverage_metrics' not in st.session_state): 
         # Se è in cache l'abbiamo caricato sopra. Pulsante per ricalcolo.
         if not metrics and comparison_text:
            with st.spinner("Calculating semantic coverage..."):
                 metrics, df_matches, _ = text_coverage.calculate_text_coverage_metrics(selected_user, source_key, texts_full, comparison_text, threshold=coverage_threshold)
                 st.session_state['coverage_metrics'] = metrics
                 st.session_state['coverage_matches'] = df_matches

    if 'coverage_metrics' in st.session_state:
        res = st.session_state['coverage_metrics']
        df = st.session_state['coverage_matches']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{res['precision']:.2f}")
        c2.metric("Recall (Coverage)", f"{res['recall']:.2f}")
        c3.metric("F1 Score", f"{res['f1']:.2f}")
        
        st.write(f"**{res['tp_recall']}** / **{res['n_full']}** posts covered.")
        
        # Analisi Sensibilità (Grafico a Linee)
        sens_data = text_coverage.sensitivity_analysis_text(texts_full, comparison_text)
        if sens_data:
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
            fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
            
            fig_sens.add_vline(x=coverage_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
            fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.error("Could not generate sensitivity data.")
        
        # Tabella Corrispondenze
        if df is not None and not df.empty:
            st.write("#### Matches")
            # Filtra elementi coperti
            st.dataframe(df[df['covered'] == True], use_container_width=True)
            

        else:
             st.warning("No detail data available for matches.")
