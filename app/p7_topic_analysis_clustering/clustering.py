import os
import json
import pandas as pd
import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.express as px
import hashlib
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "clusters")

def get_cache_path(user_id, params_sig=""):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    # Usa user_id e params_sig troncato (se troppo lungo) per nome file
    # Sanitizza params_sig per sicurezza
    clean_sig = params_sig.replace("_", "")
    filename = f"cluster_{user_id}_{clean_sig}.json"
    
    return os.path.join(CACHE_DIR, filename)

def check_cache(user_id, params_sig=""):
    path = get_cache_path(user_id, params_sig=params_sig)
    return os.path.exists(path)

def load_cache(user_id, params_sig=""):
    path = get_cache_path(user_id, params_sig=params_sig)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Ricostruisci DataFrames
            df_info = pd.DataFrame(data["topic_info"])
            
            # Ricostruisci dati visualizzazione se disponibili
            vis_data = data.get("vis_data", None)
            
            # Probs potrebbero servire come array numpy? 
            # Ritorno originale diceva 'probs': data.get("probs"), ma run_clustering ritorna array numpy di solito.
            # Ma l'app non sembra usare probs direttamente per viz, quindi lista va bene o convertiamo.
            import numpy as np
            probs = np.array(data.get("probs")) if data.get("probs") else None

            return {
                "topic_info": df_info,
                "topics": data["topics"],
                "probs": probs, 
                "vis_data": vis_data
            }, True
        except Exception as e:
            st.error(f"Error loading cache: {e}")
            return None, False
    return None, False

def save_cache(user_id, topic_info, topics, probs, vis_data, params_sig=""):
    path = get_cache_path(user_id, params_sig=params_sig)
    
    # Converti dataframe in dict
    topic_info_dict = topic_info.to_dict(orient="records")
    
    data = {
        "topic_info": topic_info_dict,
        "topics": topics,
        "probs": probs.tolist() if probs is not None else None,
        "vis_data": vis_data
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def run_clustering(user_id, texts, embeddings=None, model_name="all-mpnet-base-v2"):
    """
    Esegue clustering su testi usando UMAP + HDBSCAN (Pipeline Manuale).
    Sostituisce BERTopic per controllo completo e uso embedding specifici.
    """
    
    # 1. Embeddings
    if embeddings is None:
        st.info(f"Calculating embeddings using {model_name}...")
        sentence_model = SentenceTransformer(model_name)
        embeddings = sentence_model.encode(texts, show_progress_bar=True)
    
    # 2. Riduzione Dimensionalità (UMAP 5D per Clustering)
    st.info("Reducing dimensions (UMAP)...")
    umap_model = UMAP(
        n_neighbors=40,
        n_components=5,
        metric="cosine",
        min_dist=0.0,
        random_state=42,
    )
    umap_embeddings = umap_model.fit_transform(embeddings)

    # 3. Clustering (HDBSCAN)
    st.info("Clustering (HDBSCAN)...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=35,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topics = hdbscan_model.fit_predict(umap_embeddings)
    
    # Probabilità (soft clustering) - opzionale, HDBSCAN fornisce probabilità
    probs = hdbscan_model.probabilities_

    # 4. Estrazione Top Words per Topic (Estrazione Keyphrase)
    st.info("Extracting topic keywords...")
    docs_df = pd.DataFrame({"Doc": texts, "Topic": topics})
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    
    # Calcola approssimazione c-TF-IDF (CountVectorizer su documenti raggruppati)
    count = CountVectorizer(stop_words="english").fit(docs_per_topic.Doc.values)
    t = count.transform(docs_per_topic.Doc.values).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(len(docs_df), sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    
    # Estrai top words
    words = count.get_feature_names_out()
    topic_info_data = []
    
    for i, topic_idx in enumerate(docs_per_topic.Topic.values):
        # Ottieni indici top 10 parole
        top_indices = np.argpartition(tf_idf[:, i], -10)[-10:]
        top_words = [words[ind] for ind in top_indices]
        
        # Formato nome: ID_w1_w2_w3
        name = f"{topic_idx}_{'_'.join(top_words[:3])}"
        
        # Count
        count_val = len(docs_df[docs_df['Topic'] == topic_idx])
        
        topic_info_data.append({
             "Topic": int(topic_idx),
             "Count": int(count_val),
             "Name": name,
             "Representation": top_words,
             "CustomName": name # Redundant but safe
        })

    topic_info = pd.DataFrame(topic_info_data).sort_values("Count", ascending=False)
    
    # 5. Dati Visualizzazione (UMAP 2D)
    # Riesegui UMAP a 2D per visualizzazione
    st.info("Reducing dimensions for visualization...")
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)
    
    vis_data = {
        "x": coords_2d[:, 0].tolist(),
        "y": coords_2d[:, 1].tolist(),
        "topics": [int(t) for t in topics], # Ensure int
        "texts": [t[:100] + "..." for t in texts] # store truncated text for hover
    }
    
    # Crea una firma per i parametri
    # Firma aggiornata per riflettere pipeline manuale
    current_params_sig = "manual_umap_n40_c5_d0_s42__hdbscan_m35_predT_mpnet"
    
    # Salva risultati
    save_cache(user_id, topic_info, topics.tolist(), probs if probs is not None else None, vis_data, params_sig=current_params_sig)
    
    return {
        "topic_info": topic_info,
        "topics": topics,
        "probs": probs,
        "vis_data": vis_data
    }

def visualize_clusters(vis_data, topic_info_df=None):
    """
    Crea un grafico a dispersione Plotly dai dati di visualizzazione in cache.
    Se topic_info_df è fornito e ha 'GT_Label', usa quelli per la legenda.
    """
    if not vis_data:
        return None
        
    df_vis = pd.DataFrame({
        "x": vis_data["x"],
        "y": vis_data["y"],
        "Topic": vis_data["topics"],
        "Text": vis_data["texts"]
    })
    
    # Crea mappa etichette se topic_info_df è fornito
    label_map = {}
    if topic_info_df is not None:
        if 'GT_Label' in topic_info_df.columns:
            for _, row in topic_info_df.iterrows():
                try:
                    tid = int(row['Topic'])
                    gt = row['GT_Label']
                    score = row.get('GT_Score', 0.0)
                    # Foramto: "ID: GT Label (0.XX)"
                    label = f"{tid}: {gt} ({score:.2f})"
                    label_map[tid] = label
                except:
                    pass
        else:
             # Fallback a Name se nessuna mappatura GT
             for _, row in topic_info_df.iterrows():
                tid = int(row['Topic'])
                label_map[tid] = f"{tid}: {row.get('Name', str(tid))}"

    # Filtra outliers (Topic -1)
    df_vis = df_vis[df_vis["Topic"] != -1]
    
    # Applica etichette
    if label_map:
        df_vis["Topic Label"] = df_vis["Topic"].map(label_map).fillna(df_vis["Topic"].astype(str))
        color_col = "Topic Label"
    else:
        df_vis["Topic Label"] = df_vis["Topic"].astype(str)
        color_col = "Topic Label"
    
    fig = px.scatter(
        df_vis, 
        x="x", 
        y="y", 
        color=color_col,
        hover_data=["Text", "Topic"],
        title="Clustering dei Post (BERTopic) - (Outliers Hidden)",
        template="plotly_white"
    )
    
    # Aggiorna layout per stile pulito
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        legend_title_text="Topic (BERTopic → Ground Truth)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2"
    )
    
    # Aggiungi centroidi se possibile? Calcolo centroidi dei punti per annotazione
    if topic_info_df is not None:
        for topic_id in df_vis["Topic"].unique():
            if topic_id == -1: continue # Salta outliers per etichetta centroide solitamente
            
            # Get subset
            subset = df_vis[df_vis["Topic"] == topic_id]
            if subset.empty: continue
            
            # Media x, y
            cx = subset["x"].mean()
            cy = subset["y"].mean()
            
            fig.add_annotation(
                x=cx, y=cy,
                text=str(topic_id),
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                opacity=0.9
            )

    return fig

def map_topics_to_ground_truth(topic_info_df, ground_truth_topics, model_name="all-mpnet-base-v2"):
    """
    Mappa argomenti BERTopic (da DataFrame info) su argomenti Ground Truth usando similarità coseno.
    Restituisce un DataFrame con colonne mappatura aggiunte.
    """
    if not ground_truth_topics or topic_info_df is None or topic_info_df.empty:
        return topic_info_df
        
    # st.info("Mapping topics to Ground Truth (calculating similarities)...")
    
    # 1. Codifica Argomenti Ground Truth
    sentence_model = SentenceTransformer(model_name)
    gt_embeddings = sentence_model.encode(ground_truth_topics, show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 2. Ottieni Embeddings Argomenti
    # Itera DataFrame
    
    mapped_labels = []
    mapped_scores = []
    
    # Assicura che l'iterazione preservi l'ordine dell'indice se riassegnamo alla colonna
    for _, row in topic_info_df.iterrows():
        topic_id = row['Topic']
        
        if topic_id == -1: # Outlier
            mapped_labels.append("Outliers")
            mapped_scores.append(0.0)
            continue
            
        # Costruisci rappresentazione testo semplificata da 'Name' o parsiamo keywords se disponibili
        # 'Name' usually looks like "0_word1_word2..."
        # Or we can use the specific word columns if available in the DF structure BERTopic returns.
        # Default 'Topic' info has "Representation" column (list of words) in newer versions, or we rely on Name.
        
        # Proviamo a estrarre parole da Name se Representation non è facile
        # Formato nome: ID_w1_w2_w3
        topic_name = str(row.get('Name', ''))
        parts = topic_name.split('_')
        if len(parts) > 1:
            words = " ".join(parts[1:])
        else:
            words = topic_name
            
        # Encode
        topic_emb = sentence_model.encode(words)
        
        # Calcola Sim
        sims = cosine_similarity([topic_emb], gt_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        best_label = ground_truth_topics[best_idx]
        
        mapped_labels.append(best_label)
        mapped_scores.append(float(best_score))
        
    # Aggiungi a DF
    # Use .loc to be safe or assign new col
    new_df = topic_info_df.copy()
    new_df['GT_Label'] = mapped_labels
    new_df['GT_Score'] = mapped_scores
    
    return new_df
