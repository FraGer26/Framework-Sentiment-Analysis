import os
import json
import pandas as pd
import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.express as px
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "clusters")

def get_cache_path(user_id, params_sig=""):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    # Use user_id and truncated params_sig (if too long) for filename
    # Sanitize params_sig just in case
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
            
            # Reconstruct DataFrames
            df_info = pd.DataFrame(data["topic_info"])
            
            # Reconstruct visualization data if available
            vis_data = data.get("vis_data", None)
            
            # Probs might be needed as numpy array? 
            # Original return said 'probs': data.get("probs"), but run_clustering returns numpy array usually.
            # But the app doesn't seem to use probs directly for viz, so list is likely fine or we convert.
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
    
    # Convert dataframe to dict
    topic_info_dict = topic_info.to_dict(orient="records")
    
    data = {
        "topic_info": topic_info_dict,
        "topics": topics,
        "probs": probs.tolist() if probs is not None else None,
        "vis_data": vis_data
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

from hdbscan import HDBSCAN

def run_clustering(user_id, texts, embeddings=None, model_name="all-mpnet-base-v2"):
    """
    Runs BERTopic clustering on texts.
    If embeddings are provided, uses them to save time.
    """
    
    # 1. Embeddings
    if embeddings is None:
        st.info(f"Calculating embeddings using {model_name}...")
        sentence_model = SentenceTransformer(model_name)
        embeddings = sentence_model.encode(texts, show_progress_bar=True)
    
    # 2. BERTopic
    st.info("Fitting BERTopic model with custom parameters...")
    
    # Custom UMAP parameters as requested
    umap_model = UMAP(
        n_neighbors=40,
        n_components=5,
        metric="cosine",
        min_dist=0.0,
        random_state=42,
    )
    
    # Custom HDBSCAN parameters as requested
    hdbscan_model = HDBSCAN(
        min_cluster_size=35,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    
    topic_model = BERTopic(
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model,
        language="english", 
        calculate_probabilities=True, 
        verbose=True
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)
    
    # 3. Extract Info
    topic_info = topic_model.get_topic_info()
    
    # 4. Visualization Data (2D UMAP)
    # Re-run UMAP to 2D for visualization
    st.info("Reducing dimensions for visualization...")
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    coords_2d = umap_2d.fit_transform(embeddings)
    
    vis_data = {
        "x": coords_2d[:, 0].tolist(),
        "y": coords_2d[:, 1].tolist(),
        "topics": topics,
        "texts": [t[:100] + "..." for t in texts] # store truncated text for hover
    }
    
    # Create a signature for the parameters
    # We should update this string if we change parameters manually in the code
    # Current: UMAP(n=40,min=0,comp=5,seed=42) HDBSCAN(min=35,pred=T)
    current_params_sig = "umap_n40_c5_d0_s42__hdbscan_m35_predT"
    
    # Save results
    save_cache(user_id, topic_info, topics, probs if probs is not None else None, vis_data, params_sig=current_params_sig)
    
    return {
        "topic_info": topic_info,
        "topics": topics,
        "probs": probs,
        "vis_data": vis_data
    }

def visualize_clusters(vis_data, topic_info_df=None):
    """
    Creates a Plotly scatter plot from cached visualization data.
    If topic_info_df is provided and has 'GT_Label', it uses those for the legend.
    """
    if not vis_data:
        return None
        
    df_vis = pd.DataFrame({
        "x": vis_data["x"],
        "y": vis_data["y"],
        "Topic": vis_data["topics"],
        "Text": vis_data["texts"]
    })
    
    # Create a label map if topic_info_df is provided
    label_map = {}
    if topic_info_df is not None:
        if 'GT_Label' in topic_info_df.columns:
            for _, row in topic_info_df.iterrows():
                try:
                    tid = int(row['Topic'])
                    gt = row['GT_Label']
                    score = row.get('GT_Score', 0.0)
                    # Format: "ID: GT Label (0.XX)"
                    label = f"{tid}: {gt} ({score:.2f})"
                    label_map[tid] = label
                except:
                    pass
        else:
             # Fallback to Name if no GT mapping
             for _, row in topic_info_df.iterrows():
                tid = int(row['Topic'])
                label_map[tid] = f"{tid}: {row.get('Name', str(tid))}"

    # Filter out outliers (Topic -1)
    df_vis = df_vis[df_vis["Topic"] != -1]
    
    # Apply labels
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
    
    # Update layout to match clean style
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        legend_title_text="Topic (BERTopic → Ground Truth)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2"
    )
    
    # Add centroids if possible? 
    # Calculating centroids of points for annotation
    if topic_info_df is not None:
        for topic_id in df_vis["Topic"].unique():
            if topic_id == -1: continue # Skip outliers for centroid label usually
            
            # Get subset
            subset = df_vis[df_vis["Topic"] == topic_id]
            if subset.empty: continue
            
            # Mean x, y
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
    Maps BERTopic topics (from info DataFrame) to Ground Truth topics using cosine similarity.
    Returns a DataFrame with mapping columns added.
    """
    if not ground_truth_topics or topic_info_df is None or topic_info_df.empty:
        return topic_info_df
        
    # st.info("Mapping topics to Ground Truth (calculating similarities)...")
    
    # 1. Encode Ground Truth Topics
    sentence_model = SentenceTransformer(model_name)
    gt_embeddings = sentence_model.encode(ground_truth_topics, show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 2. Get Topic Embeddings
    # Iterate DataFrame
    
    mapped_labels = []
    mapped_scores = []
    
    # Ensure iteration preserves index order if we assign back to column
    for _, row in topic_info_df.iterrows():
        topic_id = row['Topic']
        
        if topic_id == -1: # Outlier
            mapped_labels.append("Outliers")
            mapped_scores.append(0.0)
            continue
            
        # Construct simplified text representation from 'Name' or we parse keywords if available
        # 'Name' usually looks like "0_word1_word2..."
        # Or we can use the specific word columns if available in the DF structure BERTopic returns.
        # Default 'Topic' info has "Representation" column (list of words) in newer versions, or we rely on Name.
        
        # Let's try to extract words from Name if Representation isn't easy
        # Name format: ID_w1_w2_w3
        topic_name = str(row.get('Name', ''))
        parts = topic_name.split('_')
        if len(parts) > 1:
            words = " ".join(parts[1:])
        else:
            words = topic_name
            
        # Encode
        topic_emb = sentence_model.encode(words)
        
        # Calculate Sim
        sims = cosine_similarity([topic_emb], gt_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        best_label = ground_truth_topics[best_idx]
        
        mapped_labels.append(best_label)
        mapped_scores.append(float(best_score))
        
    # Add to DF
    # Use .loc to be safe or assign new col
    new_df = topic_info_df.copy()
    new_df['GT_Label'] = mapped_labels
    new_df['GT_Score'] = mapped_scores
    
    return new_df
