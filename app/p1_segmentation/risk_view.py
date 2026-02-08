import streamlit as st
from p0_global import data
from p1_segmentation import ema, segment
import plotly.graph_objects as go
import pandas as pd

def render_risk_dashboard(user_data, risk_series, segments, half_life):
    st.subheader("Risk Score Evolution & Segmentation")
    
    if risk_series.empty:
        st.warning("Not enough data to calculate risk score.")
    else:
        # Plot
        fig = go.Figure()
        
        # 1. Curva EMA (Colorata per Segmento)
        if segments:
            # Palette vibrante
            palette = [
                '#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', 
                '#00FFFF', '#FF00FF', '#FFFF00', '#008080', '#A52A2A'
            ]
            
            for i, seg in enumerate(segments):
                # Seleziona risk_series per questo segmento
                # Includiamo la data finale per continuità
                mask = (risk_series.index >= pd.to_datetime(seg['start_date'])) & (risk_series.index <= pd.to_datetime(seg['end_date']))
                segment_series = risk_series.loc[mask]
                
                if segment_series.empty: continue

                # Ciclo attraverso la palette
                seg_color = palette[i % len(palette)]
                
                # Mostra in legenda solo per il primo segmento
                show_leg = True if i == 0 else False
                trace_name = "Risk Segments" if i == 0 else f'Segment {i+1}'
                
                fig.add_trace(go.Scatter(
                    x=segment_series.index, 
                    y=segment_series.values,
                    mode='lines',
                    name=trace_name,
                    line=dict(color=seg_color, width=3),
                    showlegend=show_leg,
                    legendgroup="segments",
                    hoverinfo='skip' # Solo visivo
                ))
        else:
            # Fallback se nessun segmento
            fig.add_trace(go.Scatter(
                x=risk_series.index, 
                y=risk_series.values,
                mode='lines',
                name='Risk Score (EMA)',
                line=dict(color='rgba(0, 100, 255, 0.5)', width=2),
                hoverinfo='x+y'
            ))
        
        # 3. Traccia Unificata Invisibile per Hover Pulito
        # Questa traccia connette tutti i segmenti ma deduplica i punti di giunzione per l'hover
        if segments:
            unified_x = []
            unified_y = []
            for i, seg in enumerate(segments):
                # Aggiungi punto iniziale solo se è il primo segmento o se c'è un gap
                # Assumendo continuità, saltiamo inizio se coincide con fine precedente.
                # Metodo più robusto: se unified_x vuoto, aggiungi inizio. Altrimenti se inizio != ultimo, aggiungi inizio.
                if not unified_x:
                    unified_x.append(seg['start_date'])
                    unified_y.append(seg['start_val'])
                elif seg['start_date'] != unified_x[-1]:
                     unified_x.append(seg['start_date'])
                     unified_y.append(seg['start_val'])
                
                # Aggiungi sempre punto finale
                unified_x.append(seg['end_date'])
                unified_y.append(seg['end_val'])
                
            fig.add_trace(go.Scatter(
                x=unified_x,
                y=unified_y,
                mode='lines',
                name='Segment Trend',
                line=dict(width=0), # Invisible line
                opacity=0,
                showlegend=False,
                hovertemplate="%{y:.4f}<extra>Segment Value</extra>", # Clean label
                hoverinfo='y' # or just rely on template
            ))

        breakpoints = []
        for seg in segments:
            breakpoints.append({'Date': seg['end_date'], 'Score': seg['end_val']})
        
        # 4. Evidenzia Breakpoint (Solo visivo)
        if breakpoints:
            bp_df = pd.DataFrame(breakpoints)
            fig.add_trace(go.Scatter(
                x=bp_df['Date'],
                y=bp_df['Score'],
                mode='markers',
                name='Breakpoints',
                marker=dict(symbol='x', size=10, color='black'),
                hoverinfo='skip' # Disabilita hover per evitare doppia etichetta con Segment Trend
            ))


        fig.update_layout(
            title=f"Risk Evolution (Half-Life: {half_life} days)",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabella Breakpoint
        st.subheader("Identified Breakpoints (Trend Changes)")
        if breakpoints:
            bp_df = pd.DataFrame(breakpoints)
            bp_df["Date"] = pd.to_datetime(bp_df["Date"]).dt.strftime("%Y-%m-%d")
            st.table(bp_df.set_index("Date"))
