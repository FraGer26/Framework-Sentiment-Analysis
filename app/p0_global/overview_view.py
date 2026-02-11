# =============================================================================
# MODULO: overview_view.py
# DESCRIZIONE: Renderizza la sezione "Panoramica" dell'utente selezionato,
#              mostrando la timeline di attività, i post recenti e i post
#              con il rischio più alto.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import plotly.graph_objects as go  # Libreria per grafici interattivi
from p0_global import data, general_statistics  # Moduli locali per dati e statistiche


def render_global_overview(user_data):
    """
    Renderizza la panoramica globale per l'utente selezionato.
    
    Mostra tre sezioni:
    1. Timeline di attività (grafico a barre dei post nel tempo)
    2. Ultimi 10 post dell'utente
    3. Top 5 post con probabilità più alta di depressione (severa e moderata)
    
    Args:
        user_data (pd.DataFrame): DataFrame con i dati dell'utente selezionato
    """
    # ==========================================================================
    # SEZIONE 1: TIMELINE ATTIVITÀ UTENTE
    # Grafico a barre che mostra quanti post ha scritto l'utente per ogni giorno
    # ==========================================================================
    st.subheader("User Activity Timeline")  # Titolo della sezione
    
    # Raggruppa i post per data e conta quanti ce ne sono per ogni giorno
    # .dt.date estrae solo la parte data (senza ora) dal timestamp
    # .size() conta il numero di righe per ogni gruppo
    daily_counts = user_data.groupby(user_data["Date"].dt.date).size()
    
    # Crea un grafico a barre con Plotly
    fig_activity = go.Figure(data=[
        go.Bar(
            x=daily_counts.index,  # Asse X: le date
            y=daily_counts.values,  # Asse Y: il numero di post
            name="Posts",  # Nome nella legenda
            marker_color='black',  # Colore delle barre: nero
            marker_line_color='black',  # Colore del bordo delle barre: nero
            marker_line_width=1  # Spessore del bordo delle barre
        )
    ])
    
    # Configura il layout del grafico
    fig_activity.update_layout(
        xaxis_title="Date",  # Etichetta asse X
        yaxis_title="Number of Posts",  # Etichetta asse Y
        template="plotly_white",  # Tema bianco per il grafico
        bargap=0.1  # Spazio tra le barre (10% della larghezza)
    )
    
    # Mostra il grafico nell'interfaccia Streamlit, occupando tutta la larghezza
    st.plotly_chart(fig_activity, use_container_width=True)
    
    # ==========================================================================
    # SEZIONE 2: POST RECENTI
    # Tabella con gli ultimi 10 post dell'utente, ordinati dal più recente
    # ==========================================================================
    st.subheader("Recent Posts")  # Titolo della sezione
    
    # Seleziona solo le colonne rilevanti, ordina per data decrescente e prende i primi 10
    st.dataframe(
        user_data[["Date", "Text", "Prob_Severe_Depressed", "Prob_Moderate_Depressed"]]
        .sort_values("Date", ascending=False)  # Ordina dal più recente
        .head(10),  # Prende i primi 10
        use_container_width=True  # Occupa tutta la larghezza disponibile
    )

    # ==========================================================================
    # SEZIONE 3: POST CON RISCHIO PIÙ ALTO
    # Due colonne affiancate: top 5 severi e top 5 moderati
    # ==========================================================================
    st.markdown("---")  # Separatore orizzontale
    st.subheader("🏆 Top Risk Posts")  # Titolo della sezione
    
    # Crea due colonne affiancate per i due tipi di rischio
    col_t1, col_t2 = st.columns(2)
    
    # Colonna sinistra: Top 5 post con depressione severa più alta
    with col_t1:
        st.markdown("#### Top 5 Severely Depressed")  # Sotto-titolo
        # Ordina per probabilità severa decrescente e prende i primi 5
        top_severe = user_data.sort_values("Prob_Severe_Depressed", ascending=False).head(5)
        # Mostra la tabella con data, probabilità e testo del post
        st.dataframe(
            top_severe[["Date", "Prob_Severe_Depressed", "Text"]],
            use_container_width=True,  # Larghezza piena
            hide_index=True  # Nasconde l'indice numerico
        )
    
    # Colonna destra: Top 5 post con depressione moderata più alta
    with col_t2:
        st.markdown("#### Top 5 Moderately Depressed")  # Sotto-titolo
        # Ordina per probabilità moderata decrescente e prende i primi 5
        top_moderate = user_data.sort_values("Prob_Moderate_Depressed", ascending=False).head(5)
        # Mostra la tabella con data, probabilità e testo del post
        st.dataframe(
            top_moderate[["Date", "Prob_Moderate_Depressed", "Text"]],
            use_container_width=True,  # Larghezza piena
            hide_index=True  # Nasconde l'indice numerico
        )
