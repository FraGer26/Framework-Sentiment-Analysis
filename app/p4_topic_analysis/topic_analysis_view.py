import streamlit as st
from p2_narrative_report import report_base, report_trajectory
from p4_topic_analysis import topic_model
import os
import json

def render_topic_analysis(selected_user, user_data, api_key):
    
    # 1. Selezione Sorgente
    topic_source = st.selectbox(
        "Select Topic Source", 
        ["Raw Posts", "Narrative Base", "Narrative Trajectory"],
        key="topic_source_select"
    )
    
    # Determina testo e chiave sorgente
    text_to_analyze = ""
    source_key = "posts" # Default
    
    if topic_source == "Raw Posts":
        text_to_analyze = "\n".join(user_data["Text"].astype(str))
        source_key = "posts"
    else:
        # Carica Narrativa
        # Carica Narrativa da cache divisa
        base_data = report_base.load_base_report(selected_user)
        traj_data_loaded = report_trajectory.load_trajectory_report(selected_user)
        
        narrative_base = ""
        narrative_traj = ""
        
        if base_data:
            narrative_base = base_data.get('base_analysis', '')
        if traj_data_loaded:
            narrative_traj = traj_data_loaded.get('trajectory_summary', '')
        
        if topic_source == "Narrative Base":
            text_to_analyze = narrative_base
            source_key = "narrative_base"
        elif topic_source == "Narrative Trajectory":
            text_to_analyze = narrative_traj
            source_key = "narrative_traj"
    
    if not text_to_analyze:
        st.warning(f"No text available for {topic_source}. Please run Narrative Analysis or check data.")
        return

    # Controlla cache per argomenti
    cache_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_{source_key}_topics.json")
    is_cached = os.path.exists(cache_path)
    
    topics_data = None
    
    if is_cached:
        st.success(f"Topic analysis for **{topic_source}** available in cache.")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                topics_data = json.load(f)
        except:
            pass
    else:
        st.info(f"No cached topics for **{topic_source}**.")
        if st.button(f"Extract Topics ({topic_source})"):
             if not api_key:
                 st.error("API Key required.")
             else:
                 with st.spinner("Extracting topics..."):
                     # Chiama topic_model.extract_topics (Firma corretta: user_id, text, api_key, source_type)
                     topics_data, _ = topic_model.extract_topics(selected_user, text_to_analyze, api_key, source_type=source_key)
                     if topics_data:
                         st.success("Topics extracted!")
                         st.rerun()
    
    if topics_data:
        col_list1, col_list2, col_list3 = st.columns(3)
        with col_list1:
            st.markdown("#### Positive Topics")
            for t in topics_data.get("positivetopics", []): st.write(f"- {t}")
        with col_list2:
           st.markdown("#### Neutral Topics")
           for t in topics_data.get("neutraltopics", []): st.write(f"- {t}")
        with col_list3:
           st.markdown("#### Negative Topics")
           for t in topics_data.get("negativetopics", []): st.write(f"- {t}")
