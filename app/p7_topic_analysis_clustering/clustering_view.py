# =============================================================================
# MODULO: clustering_view.py
# DESCRIZIONE: Renderizza la sezione di clustering avanzato nell'interfaccia.
#              Permette di eseguire il clustering dei post usando la pipeline
#              UMAP + HDBSCAN, e visualizza i risultati in un grafico scatter
#              e una tabella con le informazioni sui cluster.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
from p7_topic_analysis_clustering import clustering  # Modulo per il clustering
from p4_topic_analysis import topic_model  # Modulo per accedere ai topic ground truth


def render_clustering_section(selected_user, user_data):
    """
    Renderizza la sezione di clustering avanzato dei post.
    
    Mostra:
    1. I risultati del clustering in una tabella (con mappatura GT se disponibile)
    2. Un grafico scatter 2D dei cluster con i centroidi annotati
    
    La sezione gestisce cache, esecuzione e ri-esecuzione del clustering.
    
    Args:
        selected_user: ID dell'utente selezionato
        user_data (pd.DataFrame): DataFrame con i dati dell'utente
    """
    
    st.caption("Advanced clustering of posts using BERTopic.")
    
    # ==========================================================================
    # PREPARAZIONE TESTI
    # Converte i post in una lista di stringhe
    # ==========================================================================
    if 'user_data' in locals() and not user_data.empty and "Text" in user_data.columns:
        texts_full = user_data["Text"].astype(str).tolist()
    else:
        # Fallback: tenta comunque l'estrazione se user_data non è vuoto
        texts_full = user_data["Text"].astype(str).tolist() if not user_data.empty else []

    # Se non ci sono testi, mostra avviso
    if not texts_full:
        st.warning("No posts available for clustering.")
    else:
        # Layout a 2 colonne: controlli (stretta) e risultati (larga)
        cluster_col1, cluster_col2 = st.columns([1, 3])
        
        with cluster_col1:
            # =================================================================
            # CONTROLLI CLUSTERING
            # =================================================================
            cluster_source = "Raw Posts"  # Sorgente fissa (solo post grezzi)
            
            # Firma dei parametri per la pipeline manuale
            current_sig = "manual_umap_n40_c5_d0_s42__hdbscan_m35_predT_mpnet"
            
            # Controlla se i risultati sono in cache
            is_cluster_cached = clustering.check_cache(selected_user, params_sig=current_sig)
            should_run_cluster = False  # Flag per decidere se eseguire il clustering
            
            # Caricamento automatico dalla cache
            if is_cluster_cached and 'cluster_results' not in st.session_state:
                cluster_results, from_cache_cluster = clustering.load_cache(selected_user, params_sig=current_sig)
                if from_cache_cluster:
                    st.success("✅ Clusters loaded from cache.")
                    # Salva nello stato sessione per persistenza
                    st.session_state['cluster_results'] = cluster_results

            # Se i risultati sono nello stato sessione, li recupera
            if 'cluster_results' in st.session_state:
                cluster_results = st.session_state['cluster_results']
            
            # Pulsanti per eseguire il clustering
            if not is_cluster_cached and 'cluster_results' not in st.session_state:
                st.info("No cached clusters.")
                if st.button("Run Clustering"):
                    should_run_cluster = True  # Avvia il clustering
            elif st.button("Re-Run Clustering"):
                should_run_cluster = True  # Riesegue il clustering (sovrascrive la cache)
        
        # =================================================================
        # CARICAMENTO TOPIC GROUND TRUTH
        # Per la mappatura dei cluster ai topic estratti in precedenza
        # =================================================================
        raw_topics = []
        rt_path = os.path.join(topic_model.CACHE_DIR, f"{selected_user}_posts_topics.json")
        if os.path.exists(rt_path):
            try:
                with open(rt_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    # Combina tutti i topic in un'unica lista
                    raw_topics = d.get("positivetopics", []) + d.get("neutraltopics", []) + d.get("negativetopics", [])
            except:
                pass
        
        # =================================================================
        # ESECUZIONE CLUSTERING
        # Se richiesto, esegue la pipeline completa
        # =================================================================
        if should_run_cluster:
            with st.spinner("Running BERTopic clustering..."):
                cluster_results = clustering.run_clustering(selected_user, texts_full)
                st.session_state['cluster_results'] = cluster_results
        
        # =================================================================
        # VISUALIZZAZIONE RISULTATI
        # =================================================================
        if 'cluster_results' in locals() and cluster_results:
            st.write("### Clustering Results")
            topic_df = cluster_results["topic_info"]  # DataFrame con info cluster
            
            # Se disponibili topic ground truth, mappa i cluster
            if raw_topics:
                with st.spinner("Mapping topics to Ground Truth..."):
                    topic_df = clustering.map_topics_to_ground_truth(topic_df, raw_topics)
            
            # Mostra la tabella con le informazioni sui cluster
            st.write("#### Topic Overview")
            st.dataframe(topic_df, use_container_width=True)
            
            # Mostra il grafico scatter 2D dei cluster
            st.write("#### Cluster Visualization")
            fig_cluster = clustering.visualize_clusters(
                cluster_results["vis_data"],  # Coordinate UMAP 2D
                topic_info_df=topic_df  # Info cluster (con GT se disponibile)
            )
            if fig_cluster:
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Visualization data not available.")
