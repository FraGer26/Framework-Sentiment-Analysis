# =============================================================================
# MODULO: clustering.py
# DESCRIZIONE: Implementa il clustering dei post dell'utente usando una
#              pipeline manuale: SentenceTransformer per embedding, UMAP per
#              riduzione dimensionalità, e HDBSCAN per clustering.
#              Estrae le keyword principali per ogni cluster usando c-TF-IDF
#              e mappa i cluster ai topic ground truth tramite similarità coseno.
# =============================================================================

import os  # Libreria per operazioni su file e percorsi
import json  # Libreria per lettura/scrittura file JSON
import pandas as pd  # Libreria per manipolazione dati tabulari
import streamlit as st  # Framework per interfaccia web
from bertopic import BERTopic  # Libreria per topic modeling (disponibile)
from sentence_transformers import SentenceTransformer  # Modelli per embedding testuali
from umap import UMAP  # Algoritmo per riduzione dimensionalità
import plotly.express as px  # Libreria per grafici rapidi
import hashlib  # Libreria per generare hash (disponibile)
from hdbscan import HDBSCAN  # Algoritmo di clustering density-based
from sklearn.feature_extraction.text import CountVectorizer  # Per bag-of-words (c-TF-IDF)
import numpy as np  # Libreria per calcoli numerici

# Directory dove vengono salvati i risultati del clustering in cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "clusters")


def get_cache_path(user_id, params_sig=""):
    """
    Costruisce il percorso del file cache per il clustering di un utente.
    
    Il nome del file include la firma dei parametri per distinguere
    risultati ottenuti con configurazioni diverse.
    
    Args:
        user_id: ID dell'utente
        params_sig (str): Firma dei parametri di configurazione
        
    Returns:
        str: Percorso assoluto del file cache
    """
    # Crea la directory cache se non esiste
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Rimuove underscore dalla firma per pulizia del nome file
    clean_sig = params_sig.replace("_", "")
    filename = f"cluster_{user_id}_{clean_sig}.json"
    
    return os.path.join(CACHE_DIR, filename)


def check_cache(user_id, params_sig=""):
    """
    Verifica se i risultati del clustering sono in cache.
    
    Args:
        user_id: ID dell'utente
        params_sig (str): Firma dei parametri
        
    Returns:
        bool: True se il file cache esiste
    """
    path = get_cache_path(user_id, params_sig=params_sig)
    return os.path.exists(path)


def load_cache(user_id, params_sig=""):
    """
    Carica i risultati del clustering dalla cache.
    
    Ricostruisce le strutture dati (DataFrame, array NumPy)
    dal formato JSON salvato su disco.
    
    Args:
        user_id: ID dell'utente
        params_sig (str): Firma dei parametri
        
    Returns:
        tuple: (dizionario_risultati, bool_da_cache)
               - dizionario con: topic_info, topics, probs, vis_data
               - True se caricato con successo
    """
    path = get_cache_path(user_id, params_sig=params_sig)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Ricostruisce il DataFrame delle informazioni sui topic
            df_info = pd.DataFrame(data["topic_info"])
            
            # Ricostruisce i dati di visualizzazione (coordinate UMAP 2D)
            vis_data = data.get("vis_data", None)
            
            # Ricostruisce le probabilità come array NumPy
            import numpy as np
            probs = np.array(data.get("probs")) if data.get("probs") else None

            return {
                "topic_info": df_info,  # DataFrame con info sui cluster
                "topics": data["topics"],  # Lista di assegnazioni cluster per post
                "probs": probs,  # Probabilità di appartenenza al cluster
                "vis_data": vis_data  # Coordinate per la visualizzazione
            }, True
        except Exception as e:
            st.error(f"Error loading cache: {e}")
            return None, False
    return None, False


def save_cache(user_id, topic_info, topics, probs, vis_data, params_sig=""):
    """
    Salva i risultati del clustering nella cache su disco.
    
    Converte DataFrame e array NumPy in formati serializzabili JSON.
    
    Args:
        user_id: ID dell'utente
        topic_info (pd.DataFrame): DataFrame con info sui cluster
        topics (list): Lista di assegnazioni cluster
        probs (np.ndarray): Probabilità di appartenenza
        vis_data (dict): Dati per la visualizzazione UMAP 2D
        params_sig (str): Firma dei parametri
    """
    path = get_cache_path(user_id, params_sig=params_sig)
    
    # Converte il DataFrame in lista di dizionari per la serializzazione
    topic_info_dict = topic_info.to_dict(orient="records")
    
    # Prepara i dati per il salvataggio
    data = {
        "topic_info": topic_info_dict,  # Info cluster come lista di dict
        "topics": topics,  # Assegnazioni cluster
        "probs": probs.tolist() if probs is not None else None,  # Array → lista
        "vis_data": vis_data  # Coordinate visualizzazione
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def run_clustering(user_id, texts, embeddings=None, model_name="all-mpnet-base-v2"):
    """
    Esegue il clustering dei post usando una pipeline manuale.
    
    La pipeline si compone di 5 fasi:
    1. Embedding: SentenceTransformer converte i testi in vettori
    2. UMAP 5D: riduce la dimensionalità per il clustering
    3. HDBSCAN: identifica automaticamente i cluster
    4. c-TF-IDF: estrae le keyword principali per ogni cluster
    5. UMAP 2D: riduce a 2 dimensioni per la visualizzazione
    
    Questa pipeline sostituisce BERTopic per avere controllo completo
    su ogni fase del processo.
    
    Args:
        user_id: ID dell'utente
        texts (list): Lista dei testi dei post
        embeddings (np.ndarray, optional): Embedding pre-calcolati
        model_name (str): Nome del modello SentenceTransformer
        
    Returns:
        dict: Dizionario con: topic_info, topics, probs, vis_data
    """
    
    # ==========================================================================
    # FASE 1: EMBEDDING
    # Converte i testi in vettori numerici ad alta dimensionalità
    # ==========================================================================
    if embeddings is None:
        st.info(f"Calculating embeddings using {model_name}...")
        sentence_model = SentenceTransformer(model_name)  # Carica il modello
        embeddings = sentence_model.encode(texts, show_progress_bar=True)  # Genera embedding
    
    # ==========================================================================
    # FASE 2: RIDUZIONE DIMENSIONALITÀ (UMAP → 5D)
    # Riduce la dimensionalità degli embedding per il clustering
    # 5 dimensioni è un buon compromesso tra informazione preservata e velocità
    # ==========================================================================
    st.info("Reducing dimensions (UMAP)...")
    umap_model = UMAP(
        n_neighbors=40,  # Numero di vicini per preservare la struttura locale
        n_components=5,  # Dimensioni target per il clustering
        metric="cosine",  # Metrica di distanza per embedding testuali
        min_dist=0.0,  # Distanza minima tra punti (0 = cluster più densi)
        random_state=42,  # Seed per riproducibilità
    )
    umap_embeddings = umap_model.fit_transform(embeddings)  # Riduce le dimensioni

    # ==========================================================================
    # FASE 3: CLUSTERING (HDBSCAN)
    # Identifica automaticamente i cluster senza specificare il numero
    # HDBSCAN è density-based: trova cluster di densità variabile
    # ==========================================================================
    st.info("Clustering (HDBSCAN)...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=35,  # Dimensione minima di un cluster valido
        metric="euclidean",  # Metrica di distanza nello spazio UMAP
        cluster_selection_method="eom",  # Metodo di selezione: Excess of Mass
        prediction_data=True,  # Abilita predizioni su nuovi dati
    )
    topics = hdbscan_model.fit_predict(umap_embeddings)  # Assegna ogni post a un cluster
    
    # Estrae le probabilità di appartenenza al cluster (soft clustering)
    probs = hdbscan_model.probabilities_

    # ==========================================================================
    # FASE 4: ESTRAZIONE KEYWORD (c-TF-IDF)
    # Identifica le parole più rappresentative per ogni cluster
    # usando una variante class-based di TF-IDF
    # ==========================================================================
    st.info("Extracting topic keywords...")
    # Crea un DataFrame con i documenti raggruppati per cluster
    docs_df = pd.DataFrame({"Doc": texts, "Topic": topics})
    # Concatena tutti i documenti dello stesso cluster in un unico testo
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    
    # Calcola c-TF-IDF (class-based TF-IDF)
    # Step 1: Bag of Words (escluse stop words inglesi)
    count = CountVectorizer(stop_words="english").fit(docs_per_topic.Doc.values)
    t = count.transform(docs_per_topic.Doc.values).toarray()  # Matrice termine-documento
    
    # Step 2: Term Frequency normalizzata per classe
    w = t.sum(axis=1)  # Numero totale di parole per cluster
    tf = np.divide(t.T, w)  # TF normalizzata (termine / totale parole nel cluster)
    
    # Step 3: Inverse Document Frequency (tra cluster, non tra documenti)
    sum_t = t.sum(axis=0)  # Frequenza totale di ogni termine
    idf = np.log(np.divide(len(docs_df), sum_t)).reshape(-1, 1)  # IDF standard
    
    # Step 4: c-TF-IDF = TF * IDF
    tf_idf = np.multiply(tf, idf)
    
    # Estrae le top 10 parole per ogni cluster
    words = count.get_feature_names_out()  # Vocabolario
    topic_info_data = []
    
    for i, topic_idx in enumerate(docs_per_topic.Topic.values):
        # Trova gli indici delle 10 parole con punteggio c-TF-IDF più alto
        top_indices = np.argpartition(tf_idf[:, i], -10)[-10:]
        top_words = [words[ind] for ind in top_indices]  # Parole corrispondenti
        
        # Costruisce il nome del topic: "ID_parola1_parola2_parola3"
        name = f"{topic_idx}_{'_'.join(top_words[:3])}"
        
        # Conta il numero di post in questo cluster
        count_val = len(docs_df[docs_df['Topic'] == topic_idx])
        
        topic_info_data.append({
             "Topic": int(topic_idx),  # ID del cluster
             "Count": int(count_val),  # Numero di post nel cluster
             "Name": name,  # Nome descrittivo
             "Representation": top_words,  # Top 10 keyword
             "CustomName": name  # Nome personalizzato (uguale a Name)
        })

    # Crea il DataFrame ordinato per conteggio decrescente
    topic_info = pd.DataFrame(topic_info_data).sort_values("Count", ascending=False)
    
    # ==========================================================================
    # FASE 5: RIDUZIONE PER VISUALIZZAZIONE (UMAP → 2D)
    # Riesegue UMAP a 2 dimensioni per il grafico scatter
    # ==========================================================================
    st.info("Reducing dimensions for visualization...")
    umap_2d = UMAP(
        n_neighbors=15,  # Parametri ottimizzati per visualizzazione
        n_components=2,  # 2 dimensioni per il grafico
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    coords_2d = umap_2d.fit_transform(embeddings)  # Coordinate 2D
    
    # Prepara i dati per la visualizzazione
    vis_data = {
        "x": coords_2d[:, 0].tolist(),  # Coordinata X
        "y": coords_2d[:, 1].tolist(),  # Coordinata Y
        "topics": [int(t) for t in topics],  # Assegnazione cluster (come int)
        "texts": [t[:100] + "..." for t in texts]  # Testi troncati per hover
    }
    
    # Firma dei parametri per identificare questa configurazione in cache
    current_params_sig = "manual_umap_n40_c5_d0_s42__hdbscan_m35_predT_mpnet"
    
    # Salva i risultati nella cache
    save_cache(user_id, topic_info, topics.tolist(), probs if probs is not None else None, vis_data, params_sig=current_params_sig)
    
    return {
        "topic_info": topic_info,  # DataFrame con info cluster
        "topics": topics,  # Array assegnazioni cluster
        "probs": probs,  # Array probabilità
        "vis_data": vis_data  # Dati visualizzazione
    }


def visualize_clusters(vis_data, topic_info_df=None):
    """
    Crea un grafico scatter 2D dei cluster usando Plotly.
    
    Ogni punto rappresenta un post, colorato in base al cluster.
    Se disponibili le etichette ground truth, le usa per la legenda.
    Aggiunge annotazioni con i centroidi dei cluster.
    
    Args:
        vis_data (dict): Dati di visualizzazione con coordinate x, y
        topic_info_df (pd.DataFrame, optional): DataFrame con info cluster e GT
        
    Returns:
        plotly.Figure: Grafico scatter, o None se i dati non sono disponibili
    """
    if not vis_data:
        return None
    
    # Costruisce il DataFrame per il grafico
    df_vis = pd.DataFrame({
        "x": vis_data["x"],  # Coordinata UMAP 1
        "y": vis_data["y"],  # Coordinata UMAP 2
        "Topic": vis_data["topics"],  # ID cluster
        "Text": vis_data["texts"]  # Testo troncato per hover
    })
    
    # ==========================================================================
    # COSTRUZIONE MAPPA ETICHETTE
    # Associa ogni cluster a un'etichetta leggibile per la legenda
    # ==========================================================================
    label_map = {}
    if topic_info_df is not None:
        if 'GT_Label' in topic_info_df.columns:
            # Usa le etichette ground truth se disponibili
            for _, row in topic_info_df.iterrows():
                try:
                    tid = int(row['Topic'])
                    gt = row['GT_Label']  # Etichetta ground truth
                    score = row.get('GT_Score', 0.0)  # Score di similarità
                    # Formato: "ID: GT Label (0.XX)"
                    label = f"{tid}: {gt} ({score:.2f})"
                    label_map[tid] = label
                except:
                    pass
        else:
            # Fallback: usa il nome del cluster se non c'è mappatura GT
            for _, row in topic_info_df.iterrows():
                tid = int(row['Topic'])
                label_map[tid] = f"{tid}: {row.get('Name', str(tid))}"

    # Filtra gli outlier (Topic -1 = punti non assegnati a nessun cluster)
    df_vis = df_vis[df_vis["Topic"] != -1]
    
    # Applica le etichette al DataFrame
    if label_map:
        # Mappa gli ID cluster alle etichette leggibili
        df_vis["Topic Label"] = df_vis["Topic"].map(label_map).fillna(df_vis["Topic"].astype(str))
        color_col = "Topic Label"
    else:
        # Usa l'ID numerico come etichetta
        df_vis["Topic Label"] = df_vis["Topic"].astype(str)
        color_col = "Topic Label"
    
    # Crea il grafico scatter con Plotly Express
    fig = px.scatter(
        df_vis,
        x="x",  # Coordinata UMAP 1
        y="y",  # Coordinata UMAP 2
        color=color_col,  # Colora per cluster
        hover_data=["Text", "Topic"],  # Dati mostrati al hover
        title="Clustering dei Post (BERTopic) - (Outliers Hidden)",
        template="plotly_white"  # Tema grafico pulito
    )
    
    # Personalizza l'aspetto dei marcatori
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        legend_title_text="Topic (BERTopic → Ground Truth)",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2"
    )
    
    # ==========================================================================
    # ANNOTAZIONI CENTROIDI
    # Aggiunge etichette con l'ID del cluster al centro di ogni gruppo
    # ==========================================================================
    if topic_info_df is not None:
        for topic_id in df_vis["Topic"].unique():
            if topic_id == -1: continue  # Salta gli outlier
            
            # Calcola il centroide (media delle coordinate) del cluster
            subset = df_vis[df_vis["Topic"] == topic_id]
            if subset.empty: continue
            
            cx = subset["x"].mean()  # Media X
            cy = subset["y"].mean()  # Media Y
            
            # Aggiunge l'annotazione al grafico
            fig.add_annotation(
                x=cx, y=cy,
                text=str(topic_id),  # ID del cluster
                showarrow=False,  # Nessuna freccia
                font=dict(size=12, color="black"),
                bgcolor="white",  # Sfondo bianco
                bordercolor="black",  # Bordo nero
                borderwidth=1,
                opacity=0.9
            )

    return fig


def map_topics_to_ground_truth(topic_info_df, ground_truth_topics, model_name="all-mpnet-base-v2"):
    """
    Mappa i cluster identificati da HDBSCAN ai topic ground truth.
    
    Per ogni cluster, estrae le keyword e calcola la similarità coseno
    con i topic ground truth. Assegna a ogni cluster l'etichetta
    del topic ground truth più simile.
    
    Args:
        topic_info_df (pd.DataFrame): DataFrame con info sui cluster
        ground_truth_topics (list): Lista dei topic ground truth
        model_name (str): Nome del modello per gli embedding
        
    Returns:
        pd.DataFrame: DataFrame originale con colonne aggiuntive:
                       - 'GT_Label': etichetta ground truth più vicina
                       - 'GT_Score': punteggio di similarità
    """
    # Verifica che ci siano dati sufficienti
    if not ground_truth_topics or topic_info_df is None or topic_info_df.empty:
        return topic_info_df
    
    # ==========================================================================
    # FASE 1: EMBEDDING DEI TOPIC GROUND TRUTH
    # ==========================================================================
    sentence_model = SentenceTransformer(model_name)
    gt_embeddings = sentence_model.encode(ground_truth_topics, show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # ==========================================================================
    # FASE 2: MAPPATURA PER OGNI CLUSTER
    # ==========================================================================
    mapped_labels = []  # Lista delle etichette GT assegnate
    mapped_scores = []  # Lista dei punteggi di similarità
    
    for _, row in topic_info_df.iterrows():
        topic_id = row['Topic']
        
        # Gli outlier (Topic -1) vengono etichettati come "Outliers"
        if topic_id == -1:
            mapped_labels.append("Outliers")
            mapped_scores.append(0.0)
            continue
        
        # Estrae le keyword dal nome del cluster
        # Il formato del nome è: "ID_parola1_parola2_parola3..."
        topic_name = str(row.get('Name', ''))
        parts = topic_name.split('_')
        if len(parts) > 1:
            words = " ".join(parts[1:])  # Rimuove l'ID e unisce le parole
        else:
            words = topic_name
        
        # Genera l'embedding delle keyword del cluster
        topic_emb = sentence_model.encode(words)
        
        # Calcola la similarità coseno con tutti i topic ground truth
        sims = cosine_similarity([topic_emb], gt_embeddings)[0]
        best_idx = np.argmax(sims)  # Indice del GT più simile
        best_score = sims[best_idx]  # Punteggio di similarità
        best_label = ground_truth_topics[best_idx]  # Etichetta GT
        
        mapped_labels.append(best_label)
        mapped_scores.append(float(best_score))
    
    # Aggiunge le colonne al DataFrame
    new_df = topic_info_df.copy()
    new_df['GT_Label'] = mapped_labels  # Etichetta ground truth
    new_df['GT_Score'] = mapped_scores  # Punteggio di similarità
    
    return new_df
