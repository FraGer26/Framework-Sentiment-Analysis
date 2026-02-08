import streamlit as st
import clustering
import topic_model
import json
import os

def render_clustering_section(selected_user, user_data):
    
    st.caption("Advanced clustering of posts using BERTopic.")
    
    if 'user_data' in locals() and not user_data.empty and "Text" in user_data.columns:
         texts_full = user_data["Text"].astype(str).tolist()
    else:
         texts_full = user_data["Text"].astype(str).tolist() if not user_data.empty else []

    if not texts_full:
        st.warning("No posts available for clustering.")
    else:
        cluster_col1, cluster_col2 = st.columns([1,3])
        with cluster_col1:
            cluster_source = st.selectbox("Source Text", ["Raw Posts"], key="cluster_source")
            # Firma aggiornata per pipeline manuale
            current_sig = "manual_umap_n40_c5_d0_s42__hdbscan_m35_predT_mpnet" 
            
            is_cluster_cached = clustering.check_cache(selected_user, params_sig=current_sig)
            should_run_cluster = False
            
            if is_cluster_cached and 'cluster_results' not in st.session_state:
                cluster_results, from_cache_cluster = clustering.load_cache(selected_user, params_sig=current_sig)
                if from_cache_cluster:
                    st.success("✅ Clusters loaded from cache.")
                    st.session_state['cluster_results'] = cluster_results

            if 'cluster_results' in st.session_state:
                 cluster_results = st.session_state['cluster_results']
                 # Mostra solo messaggio successo sopra.
            
            if not is_cluster_cached and 'cluster_results' not in st.session_state:
                 st.info("No cached clusters.")
                 if st.button("Run Clustering"):
                     should_run_cluster = True
            elif st.button("Re-Run Clustering"): # Esecuzione opzionale
                 should_run_cluster = True
        
        raw_topics = []
        rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
        if os.path.exists(rt_path):
            try:
                with open(rt_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
            except:
                pass
        
        if should_run_cluster:
            with st.spinner("Running BERTopic clustering..."):
                cluster_results = clustering.run_clustering(selected_user, texts_full)
                st.session_state['cluster_results'] = cluster_results
        
        if 'cluster_results' in locals() and cluster_results:
            st.write("### Clustering Results")
            topic_df = cluster_results["topic_info"]
            
            if raw_topics:
                with st.spinner("Mapping topics to Ground Truth..."):
                     topic_df = clustering.map_topics_to_ground_truth(topic_df, raw_topics)
            
            st.write("#### Topic Overview")
            st.dataframe(topic_df, use_container_width=True)
            
            st.write("#### Cluster Visualization")
            fig_cluster = clustering.visualize_clusters(cluster_results["vis_data"], topic_info_df=topic_df)
            if fig_cluster:
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Visualization data not available.")
