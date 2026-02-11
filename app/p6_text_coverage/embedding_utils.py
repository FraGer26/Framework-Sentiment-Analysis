# =============================================================================
# MODULO: embedding_utils.py
# DESCRIZIONE: Utility per la generazione di embedding testuali.
#              Fornisce funzioni per caricare il modello SentenceTransformer
#              e per convertire testi in vettori numerici (embedding).
#              Questi embedding vengono usati per calcolare la similarità
#              semantica tra testi e topic.
# =============================================================================

import streamlit as st  # Framework per interfaccia web (usato per cache risorse)
from sentence_transformers import SentenceTransformer  # Libreria per modelli di embedding


@st.cache_resource  # Cache Streamlit per risorse: il modello viene caricato una sola volta
def get_model(model_name="nomic-ai/modernbert-embed-base"):
    """
    Carica e restituisce il modello SentenceTransformer.
    
    Usa @st.cache_resource per evitare di ricaricare il modello
    ad ogni interazione dell'utente. Il modello viene mantenuto
    in memoria per tutta la durata della sessione Streamlit.
    
    Args:
        model_name (str): Nome del modello HuggingFace da caricare
                          Default: "nomic-ai/modernbert-embed-base"
        
    Returns:
        SentenceTransformer: Istanza del modello caricato
    """
    # Carica il modello dal repository HuggingFace (o dalla cache locale)
    return SentenceTransformer(model_name)


def embed_texts(model, texts, batch_size=16):
    """
    Converte una lista di testi in embedding numerici.
    
    Gli embedding sono vettori a alta dimensionalità che rappresentano
    il significato semantico dei testi. Testi con significato simile
    avranno embedding vicini nello spazio vettoriale.
    
    Args:
        model: Istanza del modello SentenceTransformer
        texts (list): Lista di stringhe da convertire in embedding
        batch_size (int): Dimensione del batch per l'encoding (default: 16)
        
    Returns:
        np.ndarray: Matrice di embedding, shape (n_testi, dim_embedding)
    """
    return model.encode(
        texts,  # Lista dei testi da codificare
        batch_size=batch_size,  # Processa 16 testi alla volta
        convert_to_numpy=True,  # Restituisce array NumPy (non tensore PyTorch)
        normalize_embeddings=True,  # Normalizza i vettori a lunghezza unitaria (per similarità coseno)
        show_progress_bar=False,  # Non mostra la barra di progresso
    )
