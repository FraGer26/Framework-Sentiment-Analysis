# =============================================================================
# MODULO: risk_view.py
# DESCRIZIONE: Renderizza la dashboard del rischio con il grafico dell'evoluzione
#              del punteggio EMA e i breakpoint della segmentazione.
#              Ogni segmento è colorato diversamente per evidenziare
#              le fasi comportamentali dell'utente.
# =============================================================================

import streamlit as st  # Framework per l'interfaccia web
import plotly.graph_objects as go  # Libreria per grafici interattivi
import pandas as pd  # Libreria per manipolazione dati tabulari
from p0_global import data  # Modulo locale per costanti
from p1_segmentation import ema, segment  # Moduli per EMA e segmentazione


def render_risk_dashboard(user_data, risk_series, segments, half_life):
    """
    Renderizza la dashboard dell'evoluzione del rischio per un utente.
    
    Mostra:
    1. Grafico con curva EMA colorata per segmento
    2. Traccia invisibile per hover unificato
    3. Breakpoint evidenziati con marcatori 'X'
    4. Tabella dei breakpoint identificati
    
    Args:
        user_data (pd.DataFrame): DataFrame con i dati dell'utente
        risk_series (pd.Series): Serie EMA del rischio (da ema.py)
        segments (list): Lista dei segmenti (da segment.py)
        half_life (int): Periodo di dimezzamento EMA usato per il calcolo
    """
    # Titolo della sezione
    st.subheader("Risk Score Evolution & Segmentation")
    
    # Controlla se ci sono dati sufficienti per il grafico
    if risk_series.empty:
        st.warning("Not enough data to calculate risk score.")
    else:
        # ==================================================================
        # CREAZIONE GRAFICO
        # ==================================================================
        fig = go.Figure()  # Crea una figura Plotly vuota
        
        # ==================================================================
        # LIVELLO 1: CURVA EMA COLORATA PER SEGMENTO
        # Ogni segmento della serie viene disegnato con un colore diverso
        # per evidenziare visivamente le fasi comportamentali
        # ==================================================================
        if segments:
            # Palette di 10 colori vibranti per distinguere i segmenti
            palette = [
                '#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', 
                '#00FFFF', '#FF00FF', '#FFFF00', '#008080', '#A52A2A'
            ]
            
            # Per ogni segmento, disegna la porzione corrispondente della curva EMA
            for i, seg in enumerate(segments):
                # Crea una maschera booleana per selezionare i punti della serie
                # che cadono nell'intervallo temporale di questo segmento
                mask = (risk_series.index >= pd.to_datetime(seg['start_date'])) & (risk_series.index <= pd.to_datetime(seg['end_date']))
                # Estrae la sotto-serie per questo segmento
                segment_series = risk_series.loc[mask]
                
                # Salta segmenti senza dati
                if segment_series.empty: continue

                # Seleziona il colore dalla palette, ciclando se ci sono più di 10 segmenti
                seg_color = palette[i % len(palette)]
                
                # Mostra nella legenda solo il primo segmento per evitare ridondanza
                show_leg = True if i == 0 else False
                # Nome della traccia: generico per il primo, specifico per gli altri
                trace_name = "Risk Segments" if i == 0 else f'Segment {i+1}'
                
                # Aggiunge la traccia del segmento al grafico
                fig.add_trace(go.Scatter(
                    x=segment_series.index,  # Asse X: le date
                    y=segment_series.values,  # Asse Y: i valori di rischio
                    mode='lines',  # Solo linee, senza marcatori
                    name=trace_name,  # Nome nella legenda
                    line=dict(color=seg_color, width=3),  # Colore e spessore linea
                    showlegend=show_leg,  # Mostra in legenda solo il primo
                    legendgroup="segments",  # Raggruppa tutti i segmenti nella legenda
                    hoverinfo='skip'  # Disabilita hover per questa traccia (usa la unificata)
                ))
        else:
            # Se non ci sono segmenti, disegna una singola curva EMA semplice
            fig.add_trace(go.Scatter(
                x=risk_series.index,  # Asse X: le date
                y=risk_series.values,  # Asse Y: i valori
                mode='lines',  # Solo linee
                name='Risk Score (EMA)',  # Nome nella legenda
                line=dict(color='rgba(0, 100, 255, 0.5)', width=2),  # Blu semi-trasparente
                hoverinfo='x+y'  # Mostra data e valore al hover
            ))
        
        # ==================================================================
        # LIVELLO 2: TRACCIA INVISIBILE PER HOVER PULITO
        # Crea una traccia che connette i breakpoint per mostrare
        # informazioni hover pulite e deduplicate
        # ==================================================================
        if segments:
            unified_x = []  # Lista delle date dei breakpoint
            unified_y = []  # Lista dei valori ai breakpoint
            
            for i, seg in enumerate(segments):
                # Aggiunge il punto iniziale solo se è il primo segmento
                # o se c'è un gap temporale (evita duplicati nei punti di giunzione)
                if not unified_x:
                    # Primo segmento: aggiunge sempre il punto iniziale
                    unified_x.append(seg['start_date'])
                    unified_y.append(seg['start_val'])
                elif seg['start_date'] != unified_x[-1]:
                    # Se l'inizio di questo segmento è diverso dalla fine del precedente
                    unified_x.append(seg['start_date'])
                    unified_y.append(seg['start_val'])
                
                # Aggiunge sempre il punto finale del segmento
                unified_x.append(seg['end_date'])
                unified_y.append(seg['end_val'])
                
            # Aggiunge la traccia invisibile al grafico
            fig.add_trace(go.Scatter(
                x=unified_x,  # Date dei breakpoint
                y=unified_y,  # Valori ai breakpoint
                mode='lines',  # Solo linee (ma invisibili)
                name='Segment Trend',  # Nome (non mostrato)
                line=dict(width=0),  # Larghezza linea 0: invisibile
                opacity=0,  # Completamente trasparente
                showlegend=False,  # Non mostrare nella legenda
                hovertemplate="%{y:.4f}<extra>Segment Value</extra>",  # Template hover pulito
                hoverinfo='y'  # Mostra solo il valore Y
            ))

        # ==================================================================
        # LIVELLO 3: MARCATORI BREAKPOINT
        # Punti 'X' neri nei punti dove il trend cambia direzione
        # ==================================================================
        breakpoints = []  # Lista dei punti di cambio trend
        for seg in segments:
            # Ogni fine-segmento è un breakpoint (cambio di tendenza)
            breakpoints.append({'Date': seg['end_date'], 'Score': seg['end_val']})
        
        if breakpoints:
            # Converte la lista in DataFrame per Plotly
            bp_df = pd.DataFrame(breakpoints)
            # Aggiunge i marcatori 'X' al grafico
            fig.add_trace(go.Scatter(
                x=bp_df['Date'],  # Asse X: date dei breakpoint
                y=bp_df['Score'],  # Asse Y: valori ai breakpoint
                mode='markers',  # Solo marcatori, senza linee
                name='Breakpoints',  # Nome nella legenda
                marker=dict(symbol='x', size=10, color='black'),  # Simbolo X nero
                hoverinfo='skip'  # Disabilita hover per evitare doppia etichetta
            ))

        # ==================================================================
        # CONFIGURAZIONE LAYOUT DEL GRAFICO
        # ==================================================================
        fig.update_layout(
            title=f"Risk Evolution (Half-Life: {half_life} days)",  # Titolo con parametro
            xaxis_title="Date",  # Etichetta asse X
            yaxis_title="Risk Score",  # Etichetta asse Y
            template="plotly_white",  # Tema bianco pulito
            hovermode="x unified"  # Hover unificato: mostra tutti i valori alla stessa data
        )
        
        # Mostra il grafico nell'interfaccia Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # ==================================================================
        # TABELLA BREAKPOINT
        # Lista tabellare dei punti di cambio trend identificati
        # ==================================================================
        st.subheader("Identified Breakpoints (Trend Changes)")
        if breakpoints:
            # Converte in DataFrame per la visualizzazione
            bp_df = pd.DataFrame(breakpoints)
            # Formatta le date in formato leggibile (YYYY-MM-DD)
            bp_df["Date"] = pd.to_datetime(bp_df["Date"]).dt.strftime("%Y-%m-%d")
            # Mostra la tabella con la data come indice
            st.table(bp_df.set_index("Date"))
