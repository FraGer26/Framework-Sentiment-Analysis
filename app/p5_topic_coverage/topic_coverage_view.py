import streamlit as st
from p5_topic_coverage import topic_coverage
from p4_topic_analysis import topic_model
import os
import json

def render_topic_coverage(selected_user):

    st.caption("Calculate how well the Narrative Topics cover the Raw Ground Truth Topics.")
    
    # 1. Carica Argomenti Grezzi (Ground Truth)
    raw_topics = []
    rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
    if os.path.exists(rt_path):
        try:
            with open(rt_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
            
    if not raw_topics:
        st.warning("Ground Truth (Raw Posts) topics not found. Please extract them in 'Topic Analysis' first.")
        return

    # 2. Seleziona Sorgente Candidata
    candidate_source = st.selectbox("Select Candidate Topics", ["Narrative Base", "Narrative Trajectory"], key="topic_cov_cand")
    cand_key = "narrative_base" if candidate_source == "Narrative Base" else "narrative_traj"
    
    # Carica Argomenti Candidati
    cand_topics = []
    ct_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{cand_key}_topics.json")
    if os.path.exists(ct_path):
        try:
            with open(ct_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                cand_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
        except:
            pass
            
    if not cand_topics:
        st.warning(f"Candidate topics ({candidate_source}) not found. Please extract them in 'Topic Analysis' first.")
        return
        
    st.write(f"**Reference:** {len(raw_topics)} topics | **Candidate:** {len(cand_topics)} topics")
    
    cov_threshold = st.slider("Topic Similarity Threshold", 0.5, 1.0, 0.75, 0.05)
    
    # Controlla Cache
    is_cached = topic_coverage.check_cache(selected_user, cand_key, threshold=cov_threshold)
    metrics = None
    df_matches = None

    if is_cached and 'topic_metrics' not in st.session_state:
         # Nota: Usiamo variabili locali se renderizziamo immediatamente.
         metrics, df_matches, _ = topic_coverage.load_cache(selected_user, cand_key, threshold=cov_threshold)
         if metrics:
             st.success("Topic coverage results loaded from cache.")

    if st.button("Calculate Topic Coverage") or (is_cached and metrics is None):
        if not metrics:
            with st.spinner("Calculating matching..."):
                metrics, df_matches, _ = topic_coverage.calculate_coverage_metrics(selected_user, cand_key, raw_topics, cand_topics, threshold=cov_threshold)
            
    if metrics:
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{metrics['precision']:.2f}")
                c2.metric("Recall", f"{metrics['recall']:.2f}")
                c3.metric("F1 Score", f"{metrics['f1']:.2f}")
                
                # Analisi Sensibilità (Grafico a Linee)
                sens_data = topic_coverage.sensitivity_analysis(raw_topics, cand_topics)
                
                if sens_data:
                    import plotly.graph_objects as go
                    fig_sens = go.Figure()
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['f1'], mode='lines+markers', name='F1 Score'))
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['precision'], mode='lines', name='Precision', line=dict(dash='dash')))
                    fig_sens.add_trace(go.Scatter(x=sens_data['thresholds'], y=sens_data['recall'], mode='lines', name='Recall', line=dict(dash='dash')))
                    
                    fig_sens.add_vline(x=cov_threshold, line_width=1, line_dash="dash", line_color="red", annotation_text="Selected")
                    
                    fig_sens.update_layout(title="Metric Sensitivity to Threshold", xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
                    st.plotly_chart(fig_sens, use_container_width=True)
                else:
                    st.error("Could not generate sensitivity data.")

                st.write("#### Matches")
                # Mostra solo corrispondenze
                st.dataframe(df_matches[df_matches['matched'] == True], use_container_width=True)
